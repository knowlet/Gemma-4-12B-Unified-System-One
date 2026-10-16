"""Pure synthetic public artifact contracts; no frameworks or real tensor reads."""

import copy
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import jev_public_action_artifact as ART
import pytest


def manifest():
    return {
        "schema_version": 1, "kind": ART.KIND, "base_model_id": ART.MODEL_ID,
        "base_revision": ART.REVISION, "base_checkpoint": dict(ART.BASE_CHECKPOINT), "training_quantization": "none", "training_dtype": "bfloat16",
        "head_sha256": ART.HEAD_SHA, "temperature": ART.TEMPERATURE,
        "policy": "greedy_legal_action_v2", "instructions_sha256": "63874de795580b95edbcb1f7c6a6857e86f458dabba77e33814d8f08f158d68a",
        "inference_model_identity": ART.IDENTITY, "lora": copy.deepcopy(ART.LORA),
        "files": {"adapter_config.json": {"sha256": "a" * 64, "size_bytes": 1},
                  "adapter_model.safetensors": {"sha256": "b" * 64, "size_bytes": 1}},
        "training_receipt": {"path": "../receipt.json", "sha256": "c" * 64},
        "provenance": {}, "public_inputs": {}, "parent": {}, "source_handoff": {},
        "compiled_files": {}, "tensor_metadata": {}, "deployment_contract": copy.deepcopy(ART.DEPLOYMENT),
        "qualification_required": ART.QUALIFICATION_KIND,
    }


def training_receipt():
    provenance = {"train": {"ids": sorted("train-" + str(i) for i in range(120))},
                  "development": {"ids": sorted("dev-" + str(i) for i in range(22))}}
    calls = [{"split": "train", "phase": "forward", "step": i + 1, "case_id": "train-" + str(i),
              "input_shape": [1, 42]} for i in range(120)]
    calls += [{"split": "development", "phase": "evaluation", "step": 120, "case_id": name,
               "input_shape": [1, 52]} for name in provenance["development"]["ids"]]
    receipt = {"steps": 120, "seed": 42, "learning_rate": 5e-5, "model_loads": 1, "training_updates": 120, "base_checkpoint": dict(ART.BASE_CHECKPOINT),
        "temperature_before": ART.TEMPERATURE, "temperature_after": ART.TEMPERATURE,
        "training": {"completed_updates": 120, "development_examples": 22, "development_updates": 0,
            "development_loss_mean": 1.5, "elapsed_ms": 100.0, "trainable_parameters": 5193728,
            "trainable_names": ["base_model.model.model.language_model.layers." + str(layer) + ".self_attn." + head + ".lora_" + side + ".default.weight"
                for layer in range(44) for head in ("q_proj", "v_proj") for side in ("A", "B")],
            "original_head_sha256_before": ART.HEAD_SHA, "original_head_sha256_after": ART.HEAD_SHA,
            "frozen_base_versions_unchanged": True, "trace": {"backbone_calls": calls}}}
    return receipt, provenance


@pytest.mark.parametrize("blob", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b'{"a":1e999}'])
def test_finite_unique_json(blob):
    with pytest.raises(ValueError):
        ART.parse(blob)


@pytest.mark.parametrize("layout", ["snapshot-file", "repository-blob", "central-blob"])
def test_base_checkpoint_resolver_accepts_both_cache_blob_layouts_without_hashing(tmp_path, monkeypatch, layout):
    hub = tmp_path / "hub"
    repository = hub / ("models--" + ART.MODEL_ID.replace("/", "--"))
    cache = repository / "snapshots" / ART.REVISION
    cache.mkdir(parents=True)
    entry = cache / "model.safetensors"
    target = entry if layout == "snapshot-file" else repository / "blobs/digest" if layout == "repository-blob" else hub / "blobs/04/digest"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"Opaque synthetic layout fixture; not tensor data")
    if target != entry:
        entry.symlink_to("../../blobs/digest" if layout == "repository-blob" else "../../../blobs/04/digest")
    monkeypatch.setattr(ART, "pin", lambda *_args, **_kwargs: pytest.fail("Path resolution hashed checkpoint bytes"))
    assert ART.resolve_base_checkpoint(cache) == target


@pytest.mark.parametrize("escape", ["outside-hub", "sibling-model", "other-hub-directory", "central-root-link", "central-shard-link",
    "hub-link", "snapshot-link", "directory", "wrong-model", "wrong-revision"])
def test_base_checkpoint_resolver_refuses_unapproved_roots_links_and_snapshot_identity(tmp_path, monkeypatch, escape):
    hub = tmp_path / "hub"
    repository = hub / ("models--" + ART.MODEL_ID.replace("/", "--"))
    cache = repository / "snapshots" / ART.REVISION
    cache.mkdir(parents=True)
    entry = cache / "model.safetensors"
    target = tmp_path / "outside/digest"
    if escape == "sibling-model":
        target = hub / "models--unreviewed--sibling/blobs/digest"
    elif escape == "other-hub-directory":
        target = hub / "unreviewed/digest"
    elif escape == "directory":
        target = repository / "blobs/not-a-file"
        target.mkdir(parents=True)
    elif escape == "wrong-model":
        cache = hub / "models--unreviewed--sibling/snapshots" / ART.REVISION
        cache.mkdir(parents=True)
        entry = cache / "model.safetensors"
        target = cache.parents[1] / "blobs/digest"
    elif escape == "wrong-revision":
        cache = repository / "snapshots" / ("0" * 40)
        cache.mkdir()
        entry = cache / "model.safetensors"
        target = repository / "blobs/digest"
    elif escape in {"hub-link", "snapshot-link"}:
        target = repository / "blobs/digest"
    if escape != "directory":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"Opaque rejected layout fixture")
    if escape == "central-root-link":
        (hub / "blobs").symlink_to(target.parent, target_is_directory=True)
        entry.symlink_to(hub / "blobs/digest")
    elif escape == "central-shard-link":
        (hub / "blobs").mkdir()
        (hub / "blobs/04").symlink_to(target.parent, target_is_directory=True)
        entry.symlink_to(hub / "blobs/04/digest")
    else:
        entry.symlink_to(target)
    if escape == "hub-link":
        alias = tmp_path / "hub-alias"
        alias.symlink_to(hub, target_is_directory=True)
        cache = alias / repository.name / "snapshots" / ART.REVISION
    elif escape == "snapshot-link":
        actual = repository / "real-snapshot"
        cache.rename(actual)
        cache.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(ART, "pin", lambda *_args, **_kwargs: pytest.fail("Rejected path reached payload hashing"))
    reason = "Pinned model snapshot cache required" if escape in {"wrong-model", "wrong-revision"} else (
        "Ordinary confined local path required" if escape in {"hub-link", "snapshot-link"} else "Base checkpoint escapes")
    with pytest.raises(ValueError, match=reason):
        ART.resolve_base_checkpoint(cache)


@pytest.mark.parametrize("field,value", [("kind", "jev_action_lora_v1"), ("training_quantization", "nf4"),
    ("inference_model_identity", "generated-actions-nf4"), ("schema_version", True),
    ("temperature", 1.0), ("head_sha256", "0" * 64), ("qualification_required", "legacy")])
def test_public_manifest_never_relabels_legacy_or_deployment(field, value):
    candidate = manifest()
    candidate[field] = value
    with pytest.raises(ValueError):
        ART.validate_manifest(candidate)


def test_public_manifest_distinct_unqualified_and_strict_nested_types():
    candidate = manifest()
    ART.validate_manifest(candidate)
    candidate["deployment_contract"]["qualified"] = 0
    with pytest.raises(ValueError):
        ART.validate_manifest(candidate)
    candidate = manifest()
    candidate["surprise"] = 1
    with pytest.raises(ValueError):
        ART.validate_manifest(candidate)


def test_verified_public_dto_is_frozen_and_detached():
    candidate = manifest()
    value = ART.VerifiedPublicActionArtifact(ART.encode(candidate), ART.encode({"training": {"completed_updates": 120}}),
        (("adapter_config.json", b"{}"),), ART.encode({"tensor_count": 176}))
    value.manifest()["kind"] = "legacy"
    assert value.manifest()["kind"] == ART.KIND
    assert value.telemetry()["qualified"] is False
    assert value.telemetry()["task_score"] is None
    assert not hasattr(value, "decision_context")
    with pytest.raises(FrozenInstanceError):
        value.manifest_bytes = b"{}"


@pytest.mark.parametrize("mutation", ["dev-update", "dev-in-train", "missing-dev", "batch2", "head", "temperature", "step-order", "repeated-train", "float-counter", "wrong-trainables", "nonfinite-loss"])
def test_training_receipt_checks_actual_trace_and_no_dev_updates(mutation):
    receipt, provenance = training_receipt()
    ART.validate_training_receipt(receipt, provenance)
    training = receipt["training"]
    if mutation == "dev-update":
        training["development_updates"] = 1
    elif mutation == "dev-in-train":
        training["trace"]["backbone_calls"][0]["case_id"] = provenance["development"]["ids"][0]
    elif mutation == "missing-dev":
        training["trace"]["backbone_calls"].pop()
    elif mutation == "batch2":
        training["trace"]["backbone_calls"][0]["input_shape"][0] = 2
    elif mutation == "head":
        training["original_head_sha256_after"] = "0" * 64
    elif mutation == "temperature":
        receipt["temperature_after"] = 1.0
    elif mutation == "step-order":
        training["trace"]["backbone_calls"][0]["step"] = 2
    elif mutation == "repeated-train":
        training["trace"]["backbone_calls"][0]["case_id"] = training["trace"]["backbone_calls"][1]["case_id"]
    elif mutation == "float-counter":
        training["completed_updates"] = 120.0
    elif mutation == "wrong-trainables":
        training["trainable_names"][0] = training["trainable_names"][1]
    else:
        training["development_loss_mean"] = float("nan")
    with pytest.raises(ValueError):
        ART.validate_training_receipt(receipt, provenance)


def source_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(ART, "ROOT", tmp_path)
    collection = {"src/s1/unit_" + str(i) + ".py": b"# collection\n" for i in range(80)}
    collection.update({"scripts/" + name + ".py": b"# caller\n" for name in
        ("jev_browser_benchmark", "jev_local_model", "jev_public_web_cases", "jev_browser_collect_public_training")})
    kernels = {key: b"# kernel\n" for key in ART.KERNELS}
    monkeypatch.setattr(ART, "KERNELS", {key: ART.sha(value) for key, value in kernels.items()})
    for label, blob in {**collection, **kernels, **{key: b"# new\n" for key in ART.NEW_SOURCES}}.items():
        path = tmp_path / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    frozen = {"git_commit": ART.FROZEN_COMMIT, "file_count": 84,
              "files": {key: ART.pin(tmp_path / key) for key in collection}}
    path = tmp_path / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(ART.encode(frozen))
    monkeypatch.setattr(ART, "FROZEN_MANIFEST_SHA", ART.sha(path.read_bytes()))
    files = {key: ART.pin(tmp_path / key) for key in sorted(set(collection) | set(kernels) | ART.NEW_SOURCES)}
    return {"schema_id": "jev_public_action_source_handoff_v1", "collection": {
        "frozen_manifest_sha256": ART.FROZEN_MANIFEST_SHA, "git_commit": ART.FROZEN_COMMIT, "source_count": 84},
        "files": files, "identity_sha256": ART.identity(files)}


@pytest.mark.parametrize("mutation", [None, "missing-collector", "extra", "changed-kernel", "reorder", "future91", "changed-live"])
def test_source_handoff_exact90_and_separate_original84(tmp_path, monkeypatch, mutation):
    handoff = source_fixture(tmp_path, monkeypatch)
    blob = ART.encode(handoff)
    ART.verify_source_handoff(blob, expected_sha256=ART.sha(blob))
    if mutation is None:
        return
    if mutation == "missing-collector":
        del handoff["files"]["scripts/jev_browser_collect_public_training.py"]
    elif mutation == "extra":
        handoff["files"]["scripts/extra.py"] = {"sha256": "0" * 64, "size_bytes": 1}
    elif mutation == "changed-kernel":
        handoff["files"]["scripts/jev_browser_fit_actions.py"]["sha256"] = "0" * 64
    elif mutation == "reorder":
        handoff["files"] = dict(reversed(list(handoff["files"].items())))
    elif mutation == "future91":
        handoff["files"]["scripts/jev_qualify_public_actions_bf16.py"] = {"sha256": "0" * 64, "size_bytes": 1}
    else:
        (tmp_path / "src/s1/unit_0.py").write_text("# changed\n")
    handoff["identity_sha256"] = ART.identity(handoff["files"])
    blob = ART.encode(handoff)
    with pytest.raises(ValueError):
        ART.verify_source_handoff(blob, expected_sha256=ART.sha(blob))


def test_source_handoff_requires_external_blob_pin(tmp_path, monkeypatch):
    blob = ART.encode(source_fixture(tmp_path, monkeypatch))
    with pytest.raises(ValueError, match="Externally"):
        ART.verify_source_handoff(blob, expected_sha256="0" * 64)


class Array:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return copy.deepcopy(self.value)


@pytest.mark.parametrize("corrupt", [False, True])
def test_full142_compile_arrays_match_reference_without_frameworks(tmp_path, monkeypatch, corrupt):
    wires = {split: [{"case_id": split + str(i), "payload": {"model": ART.IDENTITY}} for i in range(count)] for split, count in ART.COUNTS.items()}
    cases = {split: [SimpleNamespace(id=row["case_id"]) for row in rows] for split, rows in wires.items()}
    def make_row(case):
        return {"record": {"case_id": case.id}, "inputs": {"input_ids": Array([[7, 106]]),
            "attention_mask": Array([[1, 1]]), "mm_token_type_ids": Array([[0, 0]])},
            "positions": Array([0]), "targets": Array([106])}
    references = {}
    for split in ART.COUNTS:
        references["compile-projection-" + split + ".jsonl"] = b"".join(ART.encode(w) + b"\n" for w in wires[split])
        references["compiled-" + split + ".jsonl"] = b"".join(ART.encode(ART.serialized_row(make_row(case))) + b"\n" for case in cases[split])
    admitted = SimpleNamespace(train_cases=cases["train"], development_cases=cases["development"],
        compile_wire_projection=lambda split, **_kwargs: copy.deepcopy(wires[split]))
    calls = []
    def compile_one(_model, selected, _snapshot, *, identity):
        assert identity == ART.IDENTITY
        calls.append(selected[0].id)
        row = make_row(selected[0])
        if corrupt:
            row["inputs"]["attention_mask"] = Array([[0, 1]])
        return [row]
    monkeypatch.setattr(ART, "trusted_module", lambda *_args: SimpleNamespace(compile_actions=compile_one))
    monkeypatch.setattr(ART.VerifiedPublicFitInputs, "check", lambda _self: None)
    monkeypatch.setattr(ART.VerifiedPublicFitInputs, "admitted", lambda _self: admitted)
    inputs = ART.VerifiedPublicFitInputs(tmp_path, b"{}", (), tuple(references.items()), b"{}", "a" * 64)
    gate = SimpleNamespace(check=lambda *_args: None)
    if corrupt:
        with pytest.raises(ValueError, match="Every prompt"):
            ART.compile_public_actions(object(), inputs, output=tmp_path, gate=gate)
        assert calls == ["train0"]
    else:
        rows, _pins = ART.compile_public_actions(object(), inputs, output=tmp_path, gate=gate)
        assert {k: len(v) for k, v in rows.items()} == ART.COUNTS
        assert len(calls) == 142


def recorded_gate_blob(steps=120, *, producer_pending=False):
    phases = ["before-any-project-framework-or-parent-payload", "before-public-project-admission",
        "immediately-before-legacy-parent-capture-and-Torch", "before-parent-recursive-tensor-authentication",
        "before-checkpoint-payload-hash-and-BF16-loader", "before-any-BF16-model-construction", "before-parent-injection"]
    phases += ["before-BF16-processor-compile-" + split + "-" + str(i) for split, count in ART.COUNTS.items() for i in range(count)]
    phases += ["immediately-before-unchanged-train_actions-optimizer"]
    phases += ["before-native-train-forward"] * steps + ["before-native-development-forward"] * 22
    phases += ["before-public-child-tensor-verification", "before-public-artifact-tree-or-parent-payload", "before-legacy-parent-only-verifier"]
    if not producer_pending:
        phases += ["before-public-child-tensor-verification"]
    total = 1024 * ART.GIB
    reserve = max(24 * ART.GIB, (total + 4) // 5)
    load = 2 * ART.BASE_CHECKPOINT["size_bytes"] + reserve
    rows = []
    for index, phase in enumerate(phases):
        scope = ART.resource_phase_scope(phase)
        required = {"cpu": reserve + 4 * ART.GIB, "load": load, "train": load + 30 * ART.GIB + 256 * 1024**2}[scope]
        rows.append({"sequence": index, "phase": phase, "scope": scope, "monotonic": index + 1.0,
            "expected_boot_id": "synthetic-boot", "admitted": True, "error": None,
            "swap_or_resident_credit_bytes": 0, "reserve_bytes": reserve, "required_bytes": required,
            "observation": {"MemTotal": total, "MemAvailable": total, "boot_id": "synthetic-boot"}})
    return b"".join(ART.encode(row) + b"\n" for row in rows)


@pytest.mark.parametrize("mutation", [None, "cpu-instead-of-train", "duplicate-compile", "swap-split", "eval-before-train", "missing-gate", "float-counter"])
def test_resource_journal_exact_protected_scope_and_order(mutation):
    rows = [ART.parse(line) for line in recorded_gate_blob().splitlines()]
    ART.validate_resource_evidence(recorded_gate_blob(), steps=120)
    if mutation is None:
        return
    if mutation == "cpu-instead-of-train":
        rows[5]["scope"] = "cpu"
        rows[5]["required_bytes"] = rows[5]["reserve_bytes"] + 4 * ART.GIB
    elif mutation == "duplicate-compile":
        rows[8]["phase"] = rows[7]["phase"]
    elif mutation == "swap-split":
        rows[7]["phase"] = "before-BF16-processor-compile-development-0"
    elif mutation == "eval-before-train":
        rows[150]["phase"] = "before-native-development-forward"
    elif mutation == "float-counter":
        rows[0]["reserve_bytes"] = float(rows[0]["reserve_bytes"])
    else:
        rows.pop()
    with pytest.raises(ValueError):
        ART.validate_resource_evidence(b"".join(ART.encode(row) + b"\n" for row in rows), steps=120)


@pytest.fixture
def whole_public_fixture(tmp_path, monkeypatch):
    """Tiny files keep all701/496/142 populations; only genuine runtime seams stubbed.

    The trusted public validator's token/compiler work and legacy tensor APIs are
    explicit synthetic interfaces. The new public input/material semantic paths,
    all inventories, sources, hashes, receipts, provenance and references execute.
    """
    handoff = source_fixture(tmp_path, monkeypatch)
    handoff_blob = ART.encode(handoff)
    frozen = ART.parse((tmp_path / "artifacts/jev-runtime/frozen-public-observation-36/freeze-manifest.json").read_bytes())
    public_root, preflight_root, admission_root, reference_root = (tmp_path / n for n in ("public", "preflight", "admission", "reference"))
    def put(root, label, blob):
        path = root / label
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
        return path
    wires, cases = {}, {}
    for split, count in ART.COUNTS.items():
        wires[split], cases[split] = [], []
        for index in range(count):
            name = split + "-" + str(index)
            proof = "episodes/" + name + "/raw.jsonl"
            raw_blob = ART.encode({"sequence": 0, "native": name}) + b"\n"
            put(public_root, split + "/" + proof, raw_blob)
            put(public_root, split + "/episodes/" + name + "/private.jsonl", b'{"sequence":0,"success":false}\n')
            put(public_root, split + "/episodes/" + name + "/compiled-inputs.jsonl", b'{"sequence":0,"context":8192}\n')
            payload = {"model": ART.IDENTITY, "state": {"case": name}, "questions": {
                "operation": {"criteria": {"CLICK": "click", "DONE": "done"}},
                "click_target": {"criteria": {str(i): "element" for i in range(400 if index == 0 else 2)}}}}
            wire = {"case_id": name, "native_payload": payload, "native_wire_sha256": ART.sha(ART.encode(payload)),
                "proof_refs": {"raw": {"artifact_path": proof, "sha256": ART.sha(raw_blob), "sequence": 0}}}
            wires[split].append(wire)
            cases[split].append({"id": name, "group_id": split + "-group", "split": split, "request": payload})
        put(public_root, split + "/wire_requests.jsonl", b"".join(ART.encode(w) + b"\n" for w in wires[split]))
        put(public_root, split + "/cases.jsonl", b"".join(ART.encode(c) + b"\n" for c in cases[split]))
        for label in frozen["files"]:
            put(public_root, split + "/sources/" + label, (tmp_path / label).read_bytes())
    put(public_root, "receipt.json", ART.encode({"schema_id": "jev-public-native-training-collection-v5", "stage": "complete", "all24_mandatory_pair_gate": True}))
    put(public_root, "collection_plan.json", b"{}")
    put(public_root, "exclusions.json", b'{"DONE":2,"SCROLL":1}')
    for index in range(ART.PUBLIC_FILE_COUNT - len(ART.tree_sizes(public_root))):
        put(public_root, "other/raw-" + str(index), b"retained")
    public_files = ART.inventory(public_root)
    public_inventory = {"schema_version": 1, "files": public_files, "identity_sha256": ART.identity(public_files)}
    monkeypatch.setattr(ART, "PUBLIC_IDENTITY", public_inventory["identity_sha256"])
    monkeypatch.setattr(ART, "PUBLIC_TOTAL_BYTES", sum(p["size_bytes"] for p in public_files.values()))
    wire_identities = {}
    for split, rows in wires.items():
        wire_identities[split] = {"native_ordered_wire_sha256": ART.sha(b"".join(ART.encode(w["native_payload"]) + b"\n" for w in rows)),
            "canonical_request_fingerprints": [ART.sha(json.dumps(w["native_payload"], sort_keys=True, separators=(",", ":")).encode()) for w in rows]}
    mappings = {"train": [{"disposition": "accepted-fit-row"}] * 120 + [{"disposition": "bootstrap-non-fit"}] * 44
            + [{"disposition": "excluded-no-input-DONE"}] * 2 + [{"disposition": "excluded-no-input-SCROLL"}],
        "development": [{"disposition": "accepted-fit-row"}] * 22 + [{"disposition": "bootstrap-non-fit"}] * 5}
    public_provenance = {"collection_schema_id": "jev-public-native-training-collection-v5", "provenance_fingerprint": ART.PROVENANCE_SHA,
        "observation_profile": {"profile": ART.PROFILE, "helper_source_sha256": ART.PROFILE_HELPER_SHA,
                                "effective_snapshot_js_sha256": ART.PROFILE_JS_SHA},
        "all_native_call_mapping": mappings, "wire_identities": wire_identities}
    class SyntheticSnapshot:
        def __init__(self, root):
            self.root = root
            self.blobs = {label: (root / label).read_bytes() for label in public_files}

        def manifest(self):
            return copy.deepcopy(public_inventory)

        def verify_unchanged(self):
            ART.check_inventory(self.root, public_files)
    class SyntheticAdmitted:
        def __init__(self):
            self.snapshot = SyntheticSnapshot(public_root)
            self.train_cases = tuple(SimpleNamespace(**c) for c in cases["train"])
            self.development_cases = tuple(SimpleNamespace(**c) for c in cases["development"])
            self._provenance = copy.deepcopy(public_provenance)

        @property
        def provenance(self):
            return copy.deepcopy(self._provenance)

        def wire_rows(self, split):
            return copy.deepcopy(wires[split])
    admitted = SyntheticAdmitted()
    def synthetic_admission(root, *, reviewed_source_pins, tokenizer_cache):
        assert len(reviewed_source_pins) == 84
        ART.check_inventory(root, public_files)
        for split, rows in wires.items():
            for row in rows:
                ref = row["proof_refs"]["raw"]
                assert ART.sha((root / split / ref["artifact_path"]).read_bytes()) == ref["sha256"]
        admitted.snapshot = SyntheticSnapshot(root)
        return admitted
    parent_root = tmp_path / "legacy-parent"
    config = {**ART.LORA, "base_model_name_or_path": ART.MODEL_ID, "revision": ART.REVISION, "peft_type": "LORA", "inference_mode": True}
    put(parent_root, "adapter/adapter_config.json", ART.encode(config))
    put(parent_root, "adapter/adapter_model.safetensors", b"explicit fake legacy tensor boundary")
    put(parent_root, "adapter/manifest.json", ART.encode({"kind": "jev_action_lora_v1"}))
    put(parent_root, "receipt.json", ART.encode({"training": {"completed_updates": 96, "development_updates": 0}}))
    for index in range(ART.PARENT_FILE_COUNT - 4):
        put(parent_root, "evidence/ancestor-" + str(index), b"historical")
    parent_files = ART.inventory(parent_root)
    parent_pins = {"manifest_sha256": ART.pin(parent_root / "adapter/manifest.json")["sha256"],
        "receipt_sha256": ART.pin(parent_root / "receipt.json")["sha256"],
        "weights_sha256": ART.pin(parent_root / "adapter/adapter_model.safetensors")["sha256"]}
    monkeypatch.setattr(ART, "PARENT_PINS", parent_pins)
    monkeypatch.setattr(ART, "PARENT_TOTAL_BYTES", sum(p["size_bytes"] for p in parent_files.values()))
    tensor_shapes = {name.replace(".default.", "."): [8, 1] if ".lora_A." in name else [1, 8]
        for name in training_receipt()[0]["training"]["trainable_names"]}
    tensor_meta = {"tensor_count": 176, "parameters": 5193728, "stored_dtypes": ["torch.float32"], "tensor_shapes": tensor_shapes}
    check = {"success": True, "identity_sha256": ART.identity(parent_files), "current_identity_sha256": ART.identity(parent_files),
             "changed": [], "missing": [], "added": []}
    parent_descriptor = {"schema_version": 1, "artifact_path": "inputs/initial_action_adapter", "files": parent_files,
        "identity_sha256": ART.identity(parent_files), "lineage_depth": 2,
        "parent_manifest_sha256": parent_pins["manifest_sha256"], "parent_training_receipt_sha256": parent_pins["receipt_sha256"],
        "parent_weights_sha256": parent_pins["weights_sha256"], "historical_training_cases": 521, "historical_completed_updates": 521,
        "initial_tensor_verification": None, "original_verification": check, "frozen_verification": check}
    parent_blob = ART.encode(parent_descriptor)
    put(preflight_root / "stage-01", "parent-capture.json", parent_blob)
    for label in public_files:
        put(preflight_root / "stage-01/inputs/public_collection", label, (public_root / label).read_bytes())
    framework_root = tmp_path / "framework"
    closure_files, packages, versions, origins = {}, {}, {}, {}
    distributions = {}
    for index in range(9):
        name = "synthetic-package-" + str(index)
        path = put(framework_root, name + ".dist-info/METADATA", b"Version: 1.0\n")
        entry = {"path": str(path), **ART.pin(path)}
        closure_files["distribution/" + name] = entry
        packages[name] = {"metadata_path": str(path), **ART.pin(path)}
        versions[name] = "1.0"
        distributions[name] = SimpleNamespace(_path=path.parent, version="1.0")
    for index in range(7):
        name = "synthetic_public_processor_module_" + str(index)
        path = put(framework_root, name + ".py", b"# source only, never executed\n")
        closure_files["module/" + name] = {"path": str(path), **ART.pin(path)}
        origins[name] = str(path)
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(path)))
    for index in range(51 - len(closure_files)):
        path = put(framework_root, "support/" + str(index) + ".py", b"# source only\n")
        closure_files["support/" + str(index)] = {"path": str(path), **ART.pin(path)}
    closure = {"interpreter": str(Path(sys.executable).absolute()), "python_major_minor": list(sys.version_info[:2]),
        "files": closure_files, "packages": packages, "versions": versions, "module_origins": origins, "components": {}}
    closure_blob = ART.encode(closure)
    monkeypatch.setattr(ART, "PROCESSOR_CLOSURE_SHA", ART.sha(closure_blob))
    monkeypatch.setattr(ART.importlib.metadata, "distribution", lambda name: distributions[name])
    cache = tmp_path / "artifacts/jev-local-hf/hub" / ("models--" + ART.MODEL_ID.replace("/", "--")) / "snapshots" / ART.REVISION
    metadata = {"processor_config.json": b"{}", "config.json": b"{}", "generation_config.json": b'{"eos_token_id":[1,106,50]}',
                "tokenizer.json": b"{}", "tokenizer_config.json": b"{}", "chat_template.jinja": b"synthetic"}
    for label, blob in metadata.items():
        put(cache, label, blob)
        put(reference_root, "processor-metadata/" + label, blob)
    put(reference_root, "processor-source-closure.json", closure_blob)
    put(reference_root, "resource-gates.jsonl", b"saved original compiler-only gates\n")
    for label, entry in closure_files.items():
        put(reference_root, "processor-runtime-sources/" + label, Path(entry["path"]).read_bytes())
    for label in handoff["files"]:
        put(reference_root, "sources/" + label, (tmp_path / label).read_bytes())
    compiled_files = {}
    for split, rows in wires.items():
        projection = [{"case_id": row["case_id"], "payload": row["native_payload"]} for row in rows]
        put(reference_root, "compile-projection-" + split + ".jsonl", b"".join(ART.encode(row) + b"\n" for row in projection))
        compiled = [{"record": {"case_id": row["case_id"]}, "inputs": {"input_ids": [[7, 106]], "attention_mask": [[1, 1]], "mm_token_type_ids": [[0, 0]]}, "positions": [0], "targets": [106]} for row in rows]
        path = put(reference_root, "compiled-" + split + ".jsonl", b"".join(ART.encode(row) + b"\n" for row in compiled))
        compiled_files[split] = ART.pin(path)
    compiler_receipt = {"stage": "complete", "compilation": {"wire_identities": wire_identities},
        "completed_row_files": compiled_files, "output_inventory_excluding_receipt": ART.inventory(reference_root)}
    path = put(reference_root, "compiler-receipt.json", ART.encode(compiler_receipt))
    monkeypatch.setattr(ART, "COMPILER_RECEIPT_SHA", ART.pin(path)["sha256"])
    meta_values = {"admission-receipt.json": {"stage": "complete", "admission_passed": True},
        "admission-provenance.json": public_provenance,
        "admission-plan.json": {"reviewed_source_pins": {k.removeprefix("src/"): v for k, v in frozen["files"].items()}},
        "preflight-receipt.json": {"stage": "complete", "test_mode": False, "primary_error": None, "finalization_errors": [],
            "public_original_inventory": public_inventory, "staged_evidence_files_excluding_receipt": {"parent-capture.json": {"sha256": ART.sha(parent_blob), "size_bytes": len(parent_blob)}}},
        "preflight-root-receipt.json": {"stage": "complete"}, "syntax-audit.json": {"observation_coverage_passed": True},
        "compiler-root-receipt.json": {"stage": "complete"}, "compiler-audit.json": {"passed": True}}
    metadata_paths = {name: put(admission_root, name, ART.encode(value)) for name, value in meta_values.items()}
    monkeypatch.setattr(ART, "FIXED_METADATA", {name: ART.pin(path)["sha256"] for name, path in metadata_paths.items()})
    def verify_legacy(path):
        assert ART.parse((path / "manifest.json").read_bytes())["kind"] == "jev_action_lora_v1"
        return SimpleNamespace(manifest_sha256=parent_pins["manifest_sha256"], receipt_sha256=parent_pins["receipt_sha256"],
            manifest={"files": {"adapter_model.safetensors": {"sha256": parent_pins["weights_sha256"]}}}, tensor_metadata=copy.deepcopy(tensor_meta))
    legacy_api = SimpleNamespace(verify_action_adapter=verify_legacy,
        verify_initial_action_adapter_splits=lambda _root, _provenance, **kwargs: None,
        _config=lambda parsed: ART.require(parsed == config, "Synthetic legacy config seam"))
    public_api = SimpleNamespace(AdmittedPublicCollection=SyntheticAdmitted, validate_public_training_collection=synthetic_admission)
    monkeypatch.setattr(ART, "trusted_module", lambda name, _path: {"s1.browser_public_training_validation": public_api, "s1.browser_action_adapter": legacy_api}[name])
    monkeypatch.setattr(ART, "tensor_metadata", lambda _blob, expected, **_kwargs: copy.deepcopy(tensor_meta))
    gate = SimpleNamespace(check=lambda *_args: None)
    kwargs = {"public_root": public_root, "preflight_root": preflight_root, "admission_root": admission_root,
        "reference_compile": reference_root, "metadata_paths": metadata_paths,
        "source_handoff_bytes": handoff_blob, "expected_source_handoff": ART.sha(handoff_blob),
        "expected_admission": ART.FIXED_METADATA["admission-receipt.json"],
        "expected_preflight": ART.FIXED_METADATA["preflight-root-receipt.json"], "tokenizer_cache": cache, "gate": gate}
    return SimpleNamespace(kwargs=kwargs, admitted=admitted, public_files=public_files, parent_root=parent_root,
        parent_descriptor=parent_descriptor, tensor_meta=tensor_meta, handoff=handoff, handoff_blob=handoff_blob,
        cache=cache, versions=versions, distributions=distributions, origins=origins, put=put, gate=gate)


@pytest.mark.parametrize("mutation", [None, "missing-raw", "missing-source", "profile", "disposition", "cross-split", "parent-metadata"])
def test_whole_public_fit_inputs_positive_and_semantic_refusals(whole_public_fixture, mutation):
    fixture = whole_public_fixture
    if mutation == "missing-raw":
        (fixture.kwargs["public_root"] / "train/episodes/train-0/raw.jsonl").unlink()
    elif mutation == "missing-source":
        (ART.ROOT / "src/s1/unit_0.py").unlink()
    elif mutation == "profile":
        fixture.admitted._provenance["observation_profile"]["profile"] = "composed-visible-text-v2"
    elif mutation == "disposition":
        fixture.admitted._provenance["all_native_call_mapping"]["train"][0] = {"disposition": "bootstrap-non-fit"}
    elif mutation == "cross-split":
        fixture.admitted.development_cases[0].group_id = "train-group"
    elif mutation == "parent-metadata":
        (fixture.kwargs["preflight_root"] / "stage-01/parent-capture.json").write_bytes(b"{}")
    if mutation is not None:
        with pytest.raises((ValueError, OSError)):
            ART.verify_public_fit_inputs(**fixture.kwargs)
        return
    verified = ART.verify_public_fit_inputs(**fixture.kwargs)
    assert type(verified) is ART.VerifiedPublicFitInputs
    assert len(verified.public_items) == 701
    assert len(verified.metadata()["parent_descriptor"]["files"]) == 496
    assert len(ART.parse(dict(verified.public_items)["train/wire_requests.jsonl"].splitlines()[0])["native_payload"]["questions"]["click_target"]["criteria"]) == 400
    verified.metadata()["parent_descriptor"]["files"].clear()
    assert len(verified.metadata()["parent_descriptor"]["files"]) == 496


def material_fixture(fixture, tmp_path):
    inputs = ART.verify_public_fit_inputs(**fixture.kwargs)
    meta = inputs.metadata()
    output = tmp_path / "public-child"
    for destination, items in (("inputs/public_collection", inputs.public_items), ("inputs/reference_compile", inputs.reference_items),
        ("inputs/metadata", tuple((k, v.encode()) for k, v in meta["metadata_blobs"].items())),
        ("inputs/initial_action_adapter", tuple((k, (fixture.parent_root / k).read_bytes()) for k in fixture.parent_descriptor["files"])),
        ("sources", tuple((k, (ART.ROOT / k).read_bytes()) for k in fixture.handoff["files"]))):
        for label, blob in items:
            fixture.put(output, destination + "/" + label, blob)
    fixture.put(output, "source-handoff.json", fixture.handoff_blob)
    fixture.put(output, "resource-gates.jsonl", recorded_gate_blob())
    for split in ART.COUNTS:
        fixture.put(output, "compiled-" + split + ".jsonl", dict(inputs.reference_items)["compiled-" + split + ".jsonl"])
    fixture.put(output, "adapter/adapter_config.json", (fixture.parent_root / "adapter/adapter_config.json").read_bytes())
    fixture.put(output, "adapter/adapter_model.safetensors", b"explicit fake child tensors")
    fixture.put(output, "adapter/manifest.pending.json", b"{}")
    receipt, _ = training_receipt()
    development_calls = [call for call in receipt["training"]["trace"]["backbone_calls"] if call["split"] == "development"]
    for call, case_id in zip(development_calls, meta["provenance"]["development"]["ids"], strict=True):
        call["case_id"] = case_id
    descriptor = copy.deepcopy(fixture.parent_descriptor)
    descriptor["initial_tensor_verification"] = {"success": True, "tensor_count": 176,
        "tensor_shapes": fixture.tensor_meta["tensor_shapes"], "weights_sha256": ART.PARENT_PINS["weights_sha256"],
        "all_tensors_exact": True, "all_trainables_qv_fp32": True, "frozen_base_versions_unchanged": True, "full_head_identity_version_preserved": True}
    public_inputs = {"inventory": meta["public_inventory"], "observation_profile": fixture.admitted.provenance["observation_profile"],
                    "metadata_files": meta["metadata_files"], "wire_identities": meta["wire_identities"]}
    receipt.update(kind=ART.KIND, stage="complete", error=None, qualified=False, task_score=None,
        tokenizer_cache=str(fixture.cache), versions=fixture.versions, initial_action_adapter=descriptor,
        provenance=meta["provenance"], public_inputs=public_inputs, tensor_metadata=fixture.tensor_meta)
    candidate = manifest()
    candidate.update(files={name: ART.pin(output / "adapter" / name) for name in ("adapter_config.json", "adapter_model.safetensors")},
        provenance=meta["provenance"], parent=descriptor, public_inputs=public_inputs,
        source_handoff={"path": "source-handoff.json", "sha256": ART.sha(fixture.handoff_blob), "identity_sha256": fixture.handoff["identity_sha256"]},
        compiled_files={split: ART.pin(output / ("compiled-" + split + ".jsonl")) for split in ART.COUNTS},
        tensor_metadata=fixture.tensor_meta)
    return output, candidate, receipt


def seal_material(output, candidate, receipt):
    receipt["output_inventory_excluding_receipt_and_manifest"] = {label: ART.pin(output / label) for label in ART.tree_sizes(output)
        if label not in {"receipt.json", "adapter/manifest.json"}}
    (output / "receipt.json").write_bytes(ART.encode(receipt))
    candidate["training_receipt"] = {"path": "../receipt.json", "sha256": ART.pin(output / "receipt.json")["sha256"]}
    (output / "adapter/manifest.json").write_bytes(ART.encode(candidate))
    return {"expected_manifest_sha256": ART.pin(output / "adapter/manifest.json")["sha256"],
        "expected_receipt_sha256": ART.pin(output / "receipt.json")["sha256"],
        "expected_source_handoff": ART.pin(output / "source-handoff.json")["sha256"]}


@pytest.mark.parametrize("mutation", [None, "legacy-kind", "wrong-parent", "dev-update", "missing-proof", "compiled-mask", "cached-config", "distribution-version", "loaded-module-origin", "changed-source", "parent-reference"])
def test_public_material_full_synthetic_semantic_path(whole_public_fixture, tmp_path, mutation):
    fixture = whole_public_fixture
    output, candidate, receipt = material_fixture(fixture, tmp_path)
    if mutation == "legacy-kind":
        candidate["kind"] = "jev_action_lora_v1"
    elif mutation == "wrong-parent":
        copied = output / "inputs/initial_action_adapter/evidence/ancestor-0"
        copied.write_bytes(b"HISTORICAL")
        candidate["parent"]["files"]["evidence/ancestor-0"] = ART.pin(copied)
        candidate["parent"]["identity_sha256"] = ART.identity(candidate["parent"]["files"])
    elif mutation == "dev-update":
        receipt["training"]["development_updates"] = 1
    elif mutation == "missing-proof":
        (output / "inputs/public_collection/train/episodes/train-0/raw.jsonl").unlink()
    elif mutation == "compiled-mask":
        path = output / "compiled-train.jsonl"
        rows = [ART.parse(line) for line in path.read_bytes().splitlines()]
        rows[0]["inputs"]["attention_mask"][0][0] = 0
        path.write_bytes(b"".join(ART.encode(row) + b"\n" for row in rows))
        candidate["compiled_files"]["train"] = ART.pin(path)
    elif mutation == "cached-config":
        (fixture.cache / "config.json").write_bytes(b'{"changed":true}')
    elif mutation == "distribution-version":
        fixture.distributions[next(iter(fixture.distributions))].version = "2.0"
    elif mutation == "loaded-module-origin":
        sys.modules[next(iter(fixture.origins))].__file__ = str(tmp_path / "foreign.py")
    elif mutation == "changed-source":
        (output / "sources/src/s1/unit_0.py").write_bytes(b"# forged copied source")
    elif mutation == "parent-reference":
        (output / "inputs/metadata/parent-capture.json").write_bytes(b"{}")
    kwargs = seal_material(output, candidate, receipt)
    if mutation is not None:
        with pytest.raises((ValueError, OSError)):
            ART.verify_public_action_artifact(output / "adapter", **kwargs, gate=fixture.gate)
    else:
        verified = ART.verify_public_action_artifact(output / "adapter", **kwargs, gate=fixture.gate)
        assert type(verified) is ART.VerifiedPublicActionArtifact
        assert verified.telemetry()["qualified"] is False
        assert verified.telemetry()["tensor_count"] == 176
        assert verified.telemetry()["training_steps"] == 120
