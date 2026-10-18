"""Tiny CPU gradients and source/manifest contracts; no model weights or GPU."""

import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import jev_browser_fit_cfg_actions as loop
import jev_browser_fit_public_cfg_actions as producer
import jev_public_action_artifact as old_artifact
import jev_public_cfg_action_artifact as artifact
import pytest
from jev_cfg_action_loss import compile_action_plan
from test_jev_cfg_action_loss import compiled


def case(case_id, split):
    row, payload, tokenizer = compiled()
    row["record"].update(case_id=case_id, split=split)
    return row, payload, tokenizer, compile_action_plan(row, payload, tokenizer)


def native(logits, calls):
    def run(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(logits=logits)

    return SimpleNamespace(device="cpu", lm=run)


@pytest.mark.parametrize("mode", ["A", "C"])
def test_actual_single_cpu_forward_full_positions_and_distinct_objective_gradients(mode):
    torch = pytest.importorskip("torch")
    row, _payload, _tok, plan = case("train-1", "train")
    logits = torch.linspace(-1, 1, len(plan.target_ids) * 128).reshape(1, -1, 128).requires_grad_()
    calls = []
    loss, metrics = loop.action_loss(native(logits, calls), row, plan=plan, mode=mode)
    loss.backward()
    assert len(calls) == 1 and calls[0]["use_cache"] is False
    assert torch.equal(calls[0]["logits_to_keep"], row["positions"])
    assert all(torch.equal(calls[0][key], value) for key, value in row["inputs"].items())
    expected = metrics["full_vocab_ce_mean"] if mode == "A" else metrics["allowed_branch_ce_mean"]
    assert float(loss.detach()) == expected
    assert metrics["objective_mode"] == mode and metrics["decision_target_tokens"] == 2
    assert metrics["compiled_row_sha256"] == plan.compiled_row_sha256
    forced = set(range(len(plan.target_ids))) - set(plan.decision_indices)
    if mode == "A":
        assert all(torch.count_nonzero(logits.grad[0, i]) > 0 for i in forced)
    else:
        assert all(torch.count_nonzero(logits.grad[0, i]) == 0 for i in forced)
        for index in plan.decision_indices:
            forbidden = [i for i in range(128) if i not in plan.allowed_ids[index]]
            assert torch.count_nonzero(logits.grad[0, index, forbidden]) == 0


def test_changed_plan_or_native_inputs_refused_before_forward():
    row, _payload, _tok, plan = case("train-1", "train")
    row["inputs"]["attention_mask"][0, 0] = 0
    with pytest.raises(ValueError, match="Compiled row changed"):
        loop.action_loss(native(None, []), row, plan=plan, mode="C")


@pytest.mark.parametrize("mode", ["A", "C"])
def test_every_full_vocab_position_including_forced_is_checked_for_nonfinite(mode):
    torch = pytest.importorskip("torch")
    row, _payload, _tok, plan = case("train-1", "train")
    logits = torch.zeros(1, len(plan.target_ids), 128)
    logits[0, 0, 127] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        loop.action_loss(native(logits, []), row, plan=plan, mode=mode)


def tiny_model():
    torch = pytest.importorskip("torch")

    class Backbone(torch.nn.Module):
        def forward(self, *, input_ids):
            return input_ids

    class LM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Parameter(torch.linspace(-0.1, 0.1, 128))
            self.backbone = Backbone()
            self.checkpoint = []

        def named_parameters(self):
            yield (
                "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default.weight",
                self.adapter,
            )

        def gradient_checkpointing_enable(self, **kwargs):
            self.checkpoint.append(("enable", kwargs))

        def gradient_checkpointing_disable(self):
            self.checkpoint.append(("disable", {}))

        def forward(self, *, input_ids, logits_to_keep, **kwargs):
            self.backbone(input_ids=input_ids)
            scale = torch.arange(1, len(logits_to_keep) + 1, dtype=torch.float32)
            return SimpleNamespace(logits=(scale[:, None] * self.adapter[None, :])[None, :, :])

    lm = LM()
    return SimpleNamespace(
        device="cpu",
        lm=lm,
        backbone=lm.backbone,
        head=torch.nn.Linear(128, 1, bias=False).requires_grad_(False),
    )


def test_matched_loop_same_sampler_one_forward_per_update_and_eval_no_dev_updates(tmp_path):
    rows = [case("train-" + str(i), "train") for i in range(2)]
    development = case("dev-0", "development")
    train = [r[0] for r in rows]
    dev = [development[0]]
    plans = {r[0]["record"]["case_id"]: r[3] for r in [*rows, development]}
    outcomes = {}
    for mode in ("A", "C"):
        output = tmp_path / mode
        output.mkdir()
        model = tiny_model()
        result = loop.train_actions(model, train, dev, output, plans=plans, mode=mode, steps=4)
        outcomes[mode] = result
        updates = [
            artifact.parse(line) for line in (output / "updates.jsonl").read_bytes().splitlines()
        ]
        evaluations = [
            artifact.parse(line)
            for line in (output / "development.jsonl").read_bytes().splitlines()
        ]
        assert result["completed_updates"] == 4 and result["development_updates"] == 0
        assert len(result["trace"]["backbone_calls"]) == 5
        assert result["execution"]["target_projection_positions"] == 5 * len(
            plans["dev-0"].target_ids
        )
        assert result["objective"] == loop.objective(mode)
        assert model.lm.training is False
        assert model.lm.checkpoint[0] == (
            "enable",
            {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        )
        assert len(evaluations) == 1 and evaluations[0]["case_id"] == "dev-0"
        assert all(row["backbone_forward_calls"] == 1 for row in updates)
        assert all(
            "full_vocab_ce_mean" in row and "allowed_branch_ce_mean" in row
            for row in updates + evaluations
        )
    a = outcomes["A"]["trace"]["backbone_calls"]
    c = outcomes["C"]["trace"]["backbone_calls"]
    assert [r["case_id"] for r in a] == [r["case_id"] for r in c]
    assert set(r["case_id"] for r in a[:2]) == {"train-0", "train-1"}


def test_training_failure_keeps_completed_update_and_cleanup_evidence(tmp_path):
    torch = pytest.importorskip("torch")
    train = case("train-1", "train")
    dev = case("dev-1", "development")
    plans = {"train-1": train[3], "dev-1": dev[3]}
    model = tiny_model()
    original, calls = model.lm.forward, []

    def failing(**kwargs):
        result = original(**kwargs)
        calls.append(1)
        if len(calls) == 2:
            result.logits = result.logits * float("nan")
        return result

    model.lm.forward = failing
    with pytest.raises(ValueError, match="finite") as caught:
        loop.train_actions(model, [train[0]], [dev[0]], tmp_path, plans=plans, mode="C", steps=2)
    evidence = caught.value.action_training_evidence
    assert evidence["completed_updates"] == 1 and evidence["completed_development_examples"] == 0
    assert len(evidence["trace"]["backbone_calls"]) == 2
    assert (
        artifact.parse((tmp_path / "training-failure.json").read_bytes())["completed_updates"] == 1
    )
    assert len((tmp_path / "updates.jsonl").read_bytes().splitlines()) == 1
    assert not (tmp_path / "training.json").exists()
    assert model.lm.training is False and model.lm.checkpoint[-1][0] == "disable"
    assert all(torch.isfinite(value).all() for _name, value in model.lm.named_parameters())


@pytest.mark.parametrize("change", ["missing", "extra", "wrong-row"])
def test_all_original_plans_required_before_optimizer(change, tmp_path):
    row, _payload, _tok, plan = case("train-1", "train")
    dev, _payload, _tok, dev_plan = case("dev-1", "development")
    plans = {"train-1": plan, "dev-1": dev_plan}
    if change == "missing":
        plans.pop("dev-1")
    elif change == "extra":
        plans["extra"] = plan
    else:
        plans["train-1"] = dev_plan
    with pytest.raises(ValueError, match="unique CFG plan|Compiled row changed"):
        loop.train_actions(None, [row], [dev], tmp_path, plans=plans, mode="C")


def test_actual_original90_plus_four_source_handoff_and_tamper_refusal():
    original = artifact.read(
        artifact.ROOT
        / "artifacts/jev-runtime/public-native-action-actual07-fit-20261016-01/source-handoff-03.json"
    )
    blob = artifact.build_source_handoff(original)
    handoff = artifact.verify_source_handoff(blob, expected_sha256=artifact.sha(blob))
    assert (
        len(handoff["files"]) == 94 and handoff["original_source_handoff_utf8"].encode() == original
    )
    assert (
        set(handoff["files"]) - set(artifact.parse(original)["files"]) == artifact.EXTENSION_SOURCES
    )
    for mutation in ("missing", "extra", "old-source", "extension", "original-bytes"):
        bad = copy.deepcopy(handoff)
        if mutation == "missing":
            bad["files"].pop("scripts/jev_cfg_action_loss.py")
        elif mutation == "extra":
            bad["files"]["scripts/unknown.py"] = {"sha256": "0" * 64, "size_bytes": 1}
        elif mutation in ("old-source", "extension"):
            label = (
                "scripts/jev_public_action_artifact.py"
                if mutation == "old-source"
                else "scripts/jev_browser_fit_cfg_actions.py"
            )
            bad["files"][label]["sha256"] = "0" * 64
        else:
            bad["original_source_handoff_utf8"] += " "
        bad["identity_sha256"] = artifact.identity(bad["files"])
        encoded = artifact.encode(bad)
        with pytest.raises(ValueError):
            artifact.verify_source_handoff(encoded, expected_sha256=artifact.sha(encoded))
    with pytest.raises(ValueError):
        old_artifact.verify_source_handoff(blob, expected_sha256=artifact.sha(blob))


def test_original_nested_public_admission_sees_exact_unchanged84_not_outer94():
    from s1.browser_public_training_validation import _authenticate_complete_source_inventory

    frozen = artifact.parse(
        artifact.read(
            artifact.ROOT
            / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json"
        )
    )["files"]
    reviewed = {label.removeprefix("src/"): pin for label, pin in frozen.items()}
    _authenticate_complete_source_inventory(reviewed)
    assert len(reviewed) == 84 and not (artifact.ROOT / "src/s1/jev_cfg_action_loss.py").exists()
    assert (artifact.ROOT / "scripts/jev_cfg_action_loss.py").is_file()
    polluted = {**reviewed, "s1/jev_cfg_action_loss.py": {"sha256": "0" * 64, "size_bytes": 1}}
    with pytest.raises(ValueError, match="Entire installed runtime source population"):
        _authenticate_complete_source_inventory(polluted)


def manifest(mode):
    return {
        "schema_version": 1,
        "kind": artifact.KIND,
        "base_model_id": artifact.MODEL_ID,
        "base_revision": artifact.REVISION,
        "base_checkpoint": dict(artifact.BASE_CHECKPOINT),
        "training_quantization": "none",
        "training_dtype": "bfloat16",
        "head_sha256": artifact.HEAD_SHA,
        "temperature": artifact.TEMPERATURE,
        "policy": "greedy_legal_action_v2",
        "instructions_sha256": "63874de795580b95edbcb1f7c6a6857e86f458dabba77e33814d8f08f158d68a",
        "inference_model_identity": artifact.IDENTITY,
        "lora": copy.deepcopy(artifact.LORA),
        "files": {"adapter_config.json": {}, "adapter_model.safetensors": {}},
        "training_receipt": {"path": "../receipt.json", "sha256": "0" * 64},
        "provenance": {},
        "public_inputs": {},
        "parent": {},
        "source_handoff": {},
        "compiled_files": {},
        "tensor_metadata": {},
        "deployment_contract": copy.deepcopy(artifact.DEPLOYMENT),
        "qualification_required": artifact.QUALIFICATION_KIND,
        "objective": loop.objective(mode),
        "cfg_plan_files": {
            split: {"sha256": "1" * 64, "size_bytes": 1} for split in artifact.COUNTS
        },
    }


@pytest.mark.parametrize("mode", ["A", "C"])
def test_new_manifest_cannot_be_legacy_qualification_or_implicit_objective(mode):
    value = manifest(mode)
    artifact.validate_manifest(value)
    with pytest.raises(ValueError):
        old_artifact.validate_manifest(value)
    for change in ("old-kind", "old-qual", "missing-plan", "fake-mode", "bool-size", "whole-NLL"):
        bad = copy.deepcopy(value)
        if change == "old-kind":
            bad["kind"] = old_artifact.KIND
        elif change == "old-qual":
            bad["qualification_required"] = old_artifact.QUALIFICATION_KIND
        elif change == "missing-plan":
            bad["cfg_plan_files"].pop("development")
        elif change == "bool-size":
            bad["cfg_plan_files"]["train"]["size_bytes"] = True
        elif change == "whole-NLL":
            bad["objective"]["loss"] = "whole-action-NLL"
        else:
            bad["objective"]["mode"] = "B"
        with pytest.raises(ValueError):
            artifact.validate_manifest(bad)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mode": "B"},
        {"steps": 240},
        {"steps": True},
        {"seed": 43},
        {"seed": True},
        {"learning_rate": 1e-4},
    ],
)
def test_matched_producer_refuses_drift_before_fresh_environment_or_output(
    tmp_path, monkeypatch, overrides
):
    monkeypatch.setattr(
        producer, "_fresh_environment", lambda: pytest.fail("Invalid configuration entered fitter")
    )
    args = dict(
        output=tmp_path / "new",
        source_handoff="unused",
        expected_source_handoff="unused",
        expected_boot_id="unused",
        mode="A",
    )
    args.update(overrides)
    with pytest.raises(ValueError, match="Matched A/C"):
        producer.fit_public_actions(**args)
    assert not (tmp_path / "new").exists()


def test_resource_floor_and_512_capacity_unchanged_except_explicit_optimizer_phase(tmp_path):
    from jev_browser_fit_public_actions import ResourceGate as OldGate

    def observe():
        return {
            "MemTotal": 512 * artifact.GIB,
            "MemAvailable": 500 * artifact.GIB,
            "boot_id": "test-boot",
        }

    original = OldGate(tmp_path / "old.jsonl", "test-boot", observe=observe)
    new = producer.ResourceGate(tmp_path / "new.jsonl", "test-boot", observe=observe)
    a = original.check("immediately-before-unchanged-train_actions-optimizer", "train")
    b = new.check("immediately-before-CFG-train_actions-optimizer", "train")
    for key in (
        "observation",
        "reserve_bytes",
        "required_bytes",
        "swap_or_resident_credit_bytes",
        "scope",
    ):
        assert a[key] == b[key]
    assert producer.MAX_RESOURCE_GATES == 512 and producer.MAX_TRAINING_STEPS == 336
    with pytest.raises(ValueError, match="Unknown protected"):
        artifact.resource_phase_scope("immediately-before-unchanged-train_actions-optimizer")


def test_sidecar_recompiled_from_original_request_and_journals_bound_to_actual_forward(
    tmp_path, monkeypatch
):
    import sys

    torch = pytest.importorskip("torch")
    rows = {split: case(split + "-1", split) for split in ("train", "development")}
    monkeypatch.setattr(artifact, "COUNTS", {"train": 1, "development": 1})
    tokenizer = rows["train"][2]
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer)
        ),
    )
    reference = tmp_path / "inputs/reference_compile"
    reference.mkdir(parents=True)
    (reference / "processor-source-closure.json").write_bytes(
        artifact.encode(
            {
                "components": {
                    "tokenizer": {
                        "class": type(tokenizer).__name__,
                        "path": str(Path(inspect.getfile(type(tokenizer))).resolve()),
                    }
                }
            }
        )
    )
    candidate = manifest("C")
    receipt = {
        "steps": 1,
        "objective": loop.objective("C"),
        "tokenizer_cache": str(producer.CACHE),
        "training": {"trace": {"backbone_calls": []}},
    }
    for split, (row, _payload, _tok, plan) in rows.items():
        blob = (
            artifact.encode({"case_id": row["record"]["case_id"], "plan": plan.evidence()}) + b"\n"
        )
        (tmp_path / ("cfg-plans-" + split + ".jsonl")).write_bytes(blob)
        candidate["cfg_plan_files"][split] = {"sha256": artifact.sha(blob), "size_bytes": len(blob)}
        (tmp_path / ("compiled-" + split + ".jsonl")).write_bytes(
            artifact.encode(artifact.serialized_row(row)) + b"\n"
        )
        loss, metrics = loop.action_loss(
            native(torch.zeros(1, len(plan.target_ids), 128), []), row, plan=plan, mode="C"
        )
        entry = {"case_id": row["record"]["case_id"], "step": 1, "loss": float(loss), **metrics}
        filename = "updates.jsonl" if split == "train" else "development.jsonl"
        (tmp_path / filename).write_bytes(artifact.encode(entry) + b"\n")
        receipt["training"]["trace"]["backbone_calls"].append(
            {"split": split, "case_id": entry["case_id"]}
        )
    (tmp_path / "training.json").write_bytes(artifact.encode(receipt["training"]))
    admitted = SimpleNamespace(
        compile_wire_projection=lambda split, **kwargs: [{"payload": rows[split][1]}]
    )
    artifact.validate_cfg_evidence(tmp_path, admitted, candidate, receipt)
    journal = tmp_path / "updates.jsonl"
    original_entry = artifact.parse(journal.read_bytes())
    for field, value in (
        ("objective_mode", "A"),
        ("loss", 0.0),
        ("target_tokens", True),
        ("compiled_row_sha256", "0" * 64),
    ):
        entry = copy.deepcopy(original_entry)
        entry[field] = value
        journal.write_bytes(artifact.encode(entry) + b"\n")
        with pytest.raises(ValueError):
            artifact.validate_cfg_evidence(tmp_path, admitted, candidate, receipt)
    journal.write_bytes(artifact.encode(original_entry) + b"\n")
    changed_receipt = copy.deepcopy(receipt)
    changed_receipt["training"]["trace"]["backbone_calls"][0]["case_id"] = "development-1"
    (tmp_path / "training.json").write_bytes(artifact.encode(changed_receipt["training"]))
    with pytest.raises(ValueError, match="actual native forward order"):
        artifact.validate_cfg_evidence(tmp_path, admitted, candidate, changed_receipt)
    (tmp_path / "training.json").write_bytes(artifact.encode(receipt["training"]))
    sidecar = tmp_path / "cfg-plans-train.jsonl"
    changed = artifact.parse(sidecar.read_bytes())
    changed["plan"]["allowed_ids"][1].append(28)
    sidecar.write_bytes(artifact.encode(changed) + b"\n")
    candidate["cfg_plan_files"]["train"] = artifact.pin(sidecar)
    with pytest.raises(ValueError, match="Recompiled original"):
        artifact.validate_cfg_evidence(tmp_path, admitted, candidate, receipt)
