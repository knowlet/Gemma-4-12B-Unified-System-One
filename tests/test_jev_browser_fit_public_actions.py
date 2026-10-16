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


@pytest.mark.parametrize(
    "scope,required", [("cpu", 30414139392), ("load", 73958270912), ("train", 106438961088)]
)
def test_resource_formula_exact_boundary_is_durable_before_return(tmp_path, scope, required):
    phase = {
        "cpu": "before-any-project-framework-or-parent-payload",
        "load": "before-checkpoint-payload-hash-and-BF16-loader",
        "train": "before-any-BF16-model-construction",
    }[scope]
    gate = FIT.ResourceGate(tmp_path / "gates.jsonl", "boot", observe=lambda: observation(required))
    row = gate.check(phase, scope)
    assert row["admitted"] is True
    assert row["required_bytes"] == required
    assert json.loads((tmp_path / "gates.jsonl").read_text()) == row
    refused = FIT.ResourceGate(
        tmp_path / "refused.jsonl", "boot", observe=lambda: observation(required - 1)
    )
    with pytest.raises(ValueError, match="refused"):
        refused.check(phase, scope)
    assert json.loads((tmp_path / "refused.jsonl").read_text())["admitted"] is False


def test_wrong_boot_and_journal_mutation_refuse(tmp_path):
    gate = FIT.ResourceGate(
        tmp_path / "wrong.jsonl", "different", observe=lambda: observation(MEM_TOTAL)
    )
    with pytest.raises(ValueError, match="same-boot"):
        gate.check("before-any-project-framework-or-parent-payload", "cpu")
    gate = FIT.ResourceGate(
        tmp_path / "changed.jsonl", "boot", observe=lambda: observation(MEM_TOTAL)
    )
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
        FIT.main(
            [
                "--output",
                str(tmp_path / "out"),
                "--source-handoff",
                "unused",
                "--expected-source-handoff",
                "a" * 64,
                "--expected-boot-id",
                "boot",
            ]
        )
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
        FIT.fit_public_actions(
            output=tmp_path / "out",
            source_handoff="unused",
            expected_source_handoff="a" * 64,
            expected_boot_id="boot",
        )
    receipt = json.loads((tmp_path / "out/receipt.json").read_text())
    assert receipt["stage"] == "failed"
    assert receipt["model_loads"] == receipt["training_updates"] == 0
    assert not (tmp_path / "out/adapter/manifest.json").exists()


@pytest.mark.parametrize("layout", ["repository-blob", "central-blob"])
def test_fit_cache_layout_reaches_unchanged_full_pin_only_after_fresh_load_and_train_gates(
    tmp_path, monkeypatch, layout
):
    """Only unrelated preflight is fake; real resolver/pin/gates reject tiny bytes."""
    hub = tmp_path / "hub"
    repository = hub / ("models--" + ART.MODEL_ID.replace("/", "--"))
    cache = repository / "snapshots" / ART.REVISION
    cache.mkdir(parents=True)
    target = repository / "blobs/digest" if layout == "repository-blob" else hub / "blobs/04/digest"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"Opaque synthetic data must fail the unchanged real model pin")
    (cache / "model.safetensors").symlink_to(target)
    monkeypatch.setattr(FIT, "ROOT", tmp_path)
    monkeypatch.setattr(FIT, "CACHE", cache)
    monkeypatch.setattr(FIT, "_fresh_environment", lambda: None)
    monkeypatch.setattr(
        FIT,
        "default_paths",
        lambda: {
            "public_root": tmp_path / "public",
            "preflight_root": tmp_path / "preflight",
            "reference_compile": tmp_path / "reference",
        },
    )
    handoff_path = tmp_path / "handoff.json"
    handoff_path.write_bytes(b"{}")
    monkeypatch.setattr(
        ART,
        "verify_source_handoff",
        lambda *_args, **_kwargs: {"files": {}, "identity_sha256": "a" * 64},
    )
    metadata = {
        "parent_descriptor": {"files": {}},
        "metadata_blobs": {},
        "provenance": {},
        "public_inventory": {},
        "provenance_bytes": '{"observation_profile":{}}',
        "metadata_files": {},
        "wire_identities": {},
    }
    inputs = SimpleNamespace(
        metadata=lambda: metadata, public_items=(), reference_items=(), check=lambda: None
    )
    monkeypatch.setattr(ART, "verify_public_fit_inputs", lambda **_kwargs: inputs)
    monkeypatch.setattr(ART, "copy_items", lambda *_args: None)
    monkeypatch.setattr(ART, "verify_reference_environment", lambda *_args, **_kwargs: {})
    api = SimpleNamespace(
        capture_initial_action_adapter=lambda *_args: {"descriptor": {"files": {}}}
    )

    def no_project_or_model(name, *_args):
        assert name == "s1.browser_action_adapter", (
            "Checkpoint refusal reached a model/kernel import"
        )
        return api

    monkeypatch.setattr(ART, "trusted_module", no_project_or_model)
    monkeypatch.setattr(ART, "validate_parent_capture", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(FIT, "_parent_checks", lambda _captured: None)
    monkeypatch.setattr(
        FIT,
        "_construct_model",
        lambda *_args: pytest.fail("Mismatching pin reached model construction"),
    )
    real_gate, real_pin, pins = FIT.ResourceGate, ART.pin, []
    monkeypatch.setattr(
        FIT,
        "ResourceGate",
        lambda path, boot: real_gate(path, boot, observe=lambda: observation(MEM_TOTAL)),
    )

    def record_pin(path, *, limit=ART.GIB):
        pins.append((path, limit))
        return real_pin(path, limit=limit)

    monkeypatch.setattr(ART, "pin", record_pin)
    output = tmp_path / "fit-output"
    with pytest.raises(ValueError, match="Pinned full BF16 checkpoint bytes differ"):
        FIT.fit_public_actions(
            output=output,
            source_handoff=handoff_path,
            expected_source_handoff="a" * 64,
            expected_boot_id="boot",
        )
    assert pins == [(target, FIT.BASE_BYTES)]
    assert (
        FIT.BASE_SHA == ART.BASE_CHECKPOINT["sha256"]
        and FIT.BASE_BYTES == ART.BASE_CHECKPOINT["size_bytes"]
    )
    rows = [
        json.loads(line) for line in (output / "resource-gates.jsonl").read_bytes().splitlines()
    ]
    assert [(row["phase"], row["scope"]) for row in rows[-2:]] == [
        ("before-checkpoint-payload-hash-and-BF16-loader", "load"),
        ("before-any-BF16-model-construction", "train"),
    ]
    assert all(row["admitted"] for row in rows)
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert (
        receipt["stage"] == "failed" and receipt["model_loads"] == receipt["training_updates"] == 0
    )
    assert not (output / "adapter/manifest.json").exists()


@pytest.mark.parametrize("steps", [337, 1000000, True, False, 0, -1, 336.0])
def test_invalid_steps_refuse_before_environment_output_or_model(tmp_path, monkeypatch, steps):
    def unexpected(*_args, **_kwargs):
        pytest.fail("Invalid steps reached a protected operation or output")

    monkeypatch.setattr(FIT, "_fresh_environment", unexpected)
    monkeypatch.setattr(FIT, "_construct_model", unexpected)
    monkeypatch.setattr(FIT, "ResourceGate", unexpected)
    monkeypatch.setattr(ART, "ordinary", unexpected)
    monkeypatch.setattr(ART, "trusted_module", unexpected)
    message = (
        "resource journal capacity" if type(steps) is int and steps > 336 else "Positive finite"
    )
    with pytest.raises(ValueError, match=message):
        FIT.fit_public_actions(
            output=tmp_path / "out",
            source_handoff="unused",
            expected_source_handoff="a" * 64,
            expected_boot_id="boot",
            steps=steps,
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("steps", [120, 240, 336])
def test_supported_steps_reach_environment_validation_without_creating_output(
    tmp_path, monkeypatch, steps
):
    def stop_before_environment():
        raise RuntimeError("supported configuration reached environment validation")

    monkeypatch.setattr(FIT, "_fresh_environment", stop_before_environment)
    with pytest.raises(RuntimeError, match="supported configuration"):
        FIT.fit_public_actions(
            output=tmp_path / "out",
            source_handoff="unused",
            expected_source_handoff="a" * 64,
            expected_boot_id="boot",
            steps=steps,
        )
    assert not (tmp_path / "out").exists()


def test_max_steps_complete_journal_fits_all512_phases_including_final_semantic_tensor_gate(
    tmp_path,
):
    steps = 336
    phases = [
        "before-any-project-framework-or-parent-payload",
        "before-public-project-admission",
        "immediately-before-legacy-parent-capture-and-Torch",
        "before-parent-recursive-tensor-authentication",
        "before-checkpoint-payload-hash-and-BF16-loader",
        "before-any-BF16-model-construction",
        "before-parent-injection",
    ]
    phases += [
        "before-BF16-processor-compile-" + split + "-" + str(index)
        for split, count in ART.COUNTS.items()
        for index in range(count)
    ]
    phases += ["immediately-before-unchanged-train_actions-optimizer"]
    phases += ["before-native-train-forward"] * steps + ["before-native-development-forward"] * 22
    phases += [
        "before-public-child-tensor-verification",
        "before-public-artifact-tree-or-parent-payload",
        "before-legacy-parent-only-verifier",
        "before-public-child-tensor-verification",
    ]
    assert len(phases) - steps == FIT.FIXED_RESOURCE_GATES == 176
    assert len(phases) == FIT.MAX_RESOURCE_GATES == 512
    assert steps == FIT.MAX_TRAINING_STEPS
    journal = tmp_path / "whole-success.jsonl"
    gate = FIT.ResourceGate(journal, "boot", observe=lambda: observation(MEM_TOTAL))
    for phase in phases:
        gate.check(phase, ART.resource_phase_scope(phase))
    blob = journal.read_bytes()
    assert len(blob.splitlines()) == 512
    ART.validate_resource_evidence(blob, steps=steps)
    with pytest.raises(ValueError, match="Bounded exact protected resource phase"):
        gate.check("before-public-child-tensor-verification", "cpu")
    assert journal.read_bytes() == blob


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
    return (
        output,
        {"training_receipt": {}, "provenance": {}, "tensor_metadata": {"tensor_shapes": {}}},
        {"stage": "pending", "steps": 120},
        inputs,
    )


def test_publication_seals_fresh_semantic_gates_and_rename_is_last_operation(tmp_path, monkeypatch):
    output, manifest, receipt, inputs = candidate_fixture(tmp_path, monkeypatch)
    real_replace = os.replace
    events = []

    def replace(source, target):
        events.append(str(target))
        real_replace(source, target)
        if str(target).endswith("adapter/manifest.json"):
            monkeypatch.setattr(
                ART,
                "pin",
                lambda *_args: pytest.fail("Fallible validation after marker replacement"),
            )

    monkeypatch.setattr(FIT.os, "replace", replace)
    FIT.publish_candidate(output, manifest, receipt, inputs=inputs, captured={}, gate=object())
    assert events[-1].endswith("adapter/manifest.json")
    saved = json.loads((output / "receipt.json").read_text())
    assert saved["output_inventory_excluding_receipt_and_manifest"]["resource-gates.jsonl"][
        "size_bytes"
    ] == len(b"before\nfinal semantic gate\n")
    assert json.loads((output / "adapter/manifest.json").read_text())["training_receipt"][
        "sha256"
    ] == ART.sha((output / "receipt.json").read_bytes())


@pytest.mark.parametrize("failure", ["semantic", "source", "temporary-fsync", "marker-rename"])
def test_failed_candidate_never_leaves_valid_marker_and_retains_partial(
    tmp_path, monkeypatch, failure
):
    output, manifest, receipt, inputs = candidate_fixture(tmp_path, monkeypatch)
    if failure == "semantic":
        monkeypatch.setattr(
            ART,
            "_verify_public_material",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("semantic")),
        )
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
    monkeypatch.setattr(
        ART, "tree_sizes", lambda _path: {str(i): 0 for i in range(ART.MAX_FILES - 264)}
    )
    FIT.check_cap(tmp_path)
    with pytest.raises(ValueError, match="cap"):
        FIT.check_cap(tmp_path, pending_files=1)


def test_main_returns_without_fallible_output_after_marker(monkeypatch):
    result = {"stage": "complete", "qualified": False}
    monkeypatch.setattr(FIT, "fit_public_actions", lambda **_kwargs: result)
    monkeypatch.setattr(
        "builtins.print", lambda *_args, **_kwargs: pytest.fail("Output after final marker")
    )
    assert (
        FIT.main(
            [
                "--run",
                "--output",
                "unused",
                "--source-handoff",
                "unused",
                "--expected-source-handoff",
                "a" * 64,
                "--expected-boot-id",
                "boot",
            ]
        )
        is result
    )
