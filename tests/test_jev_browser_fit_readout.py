"""Freeze browser fitting evidence and fail closed without loading model weights."""

import copy
import hashlib
import importlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC = importlib.util.spec_from_file_location(
        "jev_browser_fit_readout", ROOT / "scripts/jev_browser_fit_readout.py"
    )
    FIT = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(FIT)
    COLLECTOR = importlib.import_module("jev_browser_training_data")
finally:
    sys.path.pop(0)

UPSTREAM = {
    "repository": "https://github.com/browser-use/jev-ultrafast",
    "revision": "1231850a0bf1a0c0341fe408ef1668dbbfdfac46",
    "source_sha256": {
        "jev_ultrafast/__init__.py": "60f025b8d24465e2c0f7a3547d43b8020fd8843de7b6cb3f892ef63f5548cf63",
        "jev_ultrafast/agent.py": "aaf8b271cb0ee6643fe7beb525328ab80de8568c7ac044d02b9c6721161e7c8b",
        "jev_ultrafast/browser.py": "ee08dafa93d731bd292aa0263a578a29c8001cf22eef3c59fe44a159649155ee",
        "jev_ultrafast/model.py": "a85ada8458d23642c5c8a4095acf0565e6268f81dcda26f6308942b653e9c9f3",
        "jev_ultrafast/questions.py": "66a7054b85040b949e1757311257ea0c5a843c635f9a40c99f6d15ad660fb422",
        "jev_ultrafast/snapshot.js": "e50473501c8fb8e70f3b21866d987393e3f2315c639d638bd477d170e81ed78d",
        "jev_ultrafast/static/fixture.html": "e48e7d7b6225b215060692e0a28c3d0b59e465e4b94c1bc510d5f37ff0b8bfbe",
    },
}


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def collection(tmp_path, split, *, group="episode-source", state="Completion confirmed"):
    """A genuine wire-to-case conversion with one verified final DONE row."""
    folder = tmp_path / split
    folder.mkdir()
    goal = "Mark complete and stop when Completion confirmed is visibly shown."
    html = f"<!doctype html><title>{group}</title><p>Completion confirmed</p>".encode()
    fixture_sha = digest(html)
    group_id = "browser-development:" + fixture_sha
    episode_id = f"{split}-91023-0000-completion_toggle-fresh"
    payload = {
        "model": "development-teacher-request",
        "state": {"page": {"text": state, "title": group}, "history": []},
        "questions": {
            "operation": {
                "type": "choice",
                "instructions": {"goal": goal, "rules": "Require current completion evidence."},
                "criteria": {"CLICK": "Advance a pending requirement", "DONE": "Goal confirmed"},
            },
            "click_target": {
                "type": "choice",
                "instructions": {"goal": goal, "operation": "CLICK"},
                "criteria": {"1": {"element": "[1] Mark complete", "role": "button"}},
            },
        },
    }
    choice, accepted = {"operation": "DONE", "target": None}, {"DONE": []}
    row = COLLECTOR.training_case(
        payload,
        case_id=episode_id + "-step-000",
        group_id=group_id,
        task_id="completion_toggle",
        split=split,
        selected=choice,
        valid_targets=accepted,
    )
    case_bytes = (row.model_dump_json() + "\n").encode()
    wire = {
        "case_id": row.id,
        "payload": payload,
        "label_source": "scripted development teacher, fresh observed candidate IDs",
        "teacher_choice": choice,
        "valid_next_targets": accepted,
    }
    wire_bytes = (json.dumps(wire, ensure_ascii=False) + "\n").encode()
    path = folder / "cases.jsonl"
    path.write_bytes(case_bytes)
    (folder / "wire_requests.jsonl").write_bytes(wire_bytes)
    (folder / "fixtures").mkdir()
    (folder / "fixtures" / "0000.html").write_bytes(html)
    receipt = {
        "schema_version": 1,
        "stage": "complete",
        "seed": 91023,
        "repeats": 1,
        "split": split,
        "training_cases": 1,
        "scope": "development-only scripted training collection; no model evaluation",
        "upstream": copy.deepcopy(UPSTREAM),
        "collector_sha256": digest(Path(COLLECTOR.__file__).read_bytes()),
        "dataset_sha256": digest(case_bytes),
        "wire_sha256": digest(wire_bytes),
        "text_source": "scripted development labels only, never evaluation text helper",
        "episodes": [
            {
                "episode_id": episode_id,
                "task_id": "completion_toggle",
                "group_id": group_id,
                "steps": 1,
                "verified_dom": {"success": True, "mismatches": {}},
                "scope": "scripted training collection, no model score",
                "initialization": "fresh page",
                "fixture_sha256": fixture_sha,
                "goal_sha256": digest(goal.encode()),
            }
        ],
    }
    (folder / "receipt.json").write_text(json.dumps(receipt) + "\n")
    return path


def rewrite_receipt(path, change):
    receipt_path = path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    change(receipt)
    receipt_path.write_text(json.dumps(receipt) + "\n")


def test_realistic_collection_validation_preserves_receipt(tmp_path):
    path = collection(tmp_path, "train")
    before = (path.parent / "receipt.json").read_bytes()
    cases, receipt = FIT.validated_collection(path, "train")
    assert len(cases) == 1 and cases[0].gold == {"operation": "DONE"}
    assert receipt == json.loads(before)
    assert (path.parent / "receipt.json").read_bytes() == before


@pytest.mark.parametrize("filename", ["cases.jsonl", "wire_requests.jsonl", "fixtures/0000.html"])
def test_changed_collection_file_is_rejected_before_fitting(tmp_path, filename):
    path = collection(tmp_path, "train")
    changed = path.parent / filename
    changed.write_bytes(changed.read_bytes() + b" ")
    with pytest.raises(ValueError):
        FIT.validated_collection(path, "train")


@pytest.mark.parametrize("fixture_change", ["missing", "extra"])
def test_complete_fixture_inventory_is_required(tmp_path, fixture_change):
    path = collection(tmp_path, "train")
    fixtures = path.parent / "fixtures"
    if fixture_change == "missing":
        (fixtures / "0000.html").unlink()
    else:
        (fixtures / "0001.html").write_bytes((fixtures / "0000.html").read_bytes())
    with pytest.raises(ValueError):
        FIT.validated_collection(path, "train")


@pytest.mark.parametrize("field", ["goal_sha256", "group_id", "steps", "episode_id"])
def test_episode_identity_must_bind_goal_fixture_and_exact_steps(tmp_path, field):
    path = collection(tmp_path, "train")

    def change(receipt):
        receipt["episodes"][0][field] = 2 if field == "steps" else "incorrect"

    rewrite_receipt(path, change)
    with pytest.raises(ValueError):
        FIT.validated_collection(path, "train")


@pytest.mark.parametrize("corruption", ["wording", "candidate_order"])
def test_rehashed_case_still_requires_exact_wire_wording_and_candidate_order(tmp_path, corruption):
    path = collection(tmp_path, "train")
    row = json.loads(path.read_text())
    question = row["request"]["questions"][0]
    if corruption == "wording":
        question["instructions"] = question["instructions"].replace("current", "invented")
    else:
        question["criteria"] = dict(reversed(list(question["criteria"].items())))
    path.write_text(json.dumps(row) + "\n")
    rewrite_receipt(path, lambda receipt: receipt.update(dataset_sha256=digest(path.read_bytes())))
    with pytest.raises(ValueError):
        FIT.validated_collection(path, "train")


def test_rehashed_wire_still_requires_matching_training_targets(tmp_path):
    path = collection(tmp_path, "train")
    wire_path = path.parent / "wire_requests.jsonl"
    wire = json.loads(wire_path.read_text())
    wire["valid_next_targets"] = {"CLICK": ["1"]}
    wire_path.write_text(json.dumps(wire) + "\n")
    rewrite_receipt(
        path, lambda receipt: receipt.update(wire_sha256=digest(wire_path.read_bytes()))
    )
    with pytest.raises(ValueError):
        FIT.validated_collection(path, "train")


def test_episode_group_overlap_blocks_fit_before_creating_output_or_loading_model(tmp_path):
    train = collection(tmp_path, "train", state="First observation")
    development = collection(tmp_path, "development", state="Different observation")
    output = tmp_path / "learned"
    with pytest.raises(ValueError, match="overlap"):
        FIT.fit(train_cases=train, development_cases=development, output_dir=output)
    assert not output.exists()


def test_test_split_cannot_be_used_as_training_collection(tmp_path):
    path = collection(tmp_path, "test")
    with pytest.raises(ValueError):
        FIT.validated_collection(path, "train")


def test_freezing_preserves_exact_original_bytes_before_later_edits(tmp_path):
    path = collection(tmp_path, "train")
    _cases, original_receipt, snapshot = FIT._validated_snapshot(path, "train")
    frozen = tmp_path / "frozen"
    FIT._freeze_collection(snapshot, frozen)
    for relative, blob in snapshot["blobs"].items():
        assert (frozen / relative).read_bytes() == blob
        assert snapshot["manifest"]["files"][relative] == {
            "sha256": digest(blob),
            "size_bytes": len(blob),
        }
    assert json.loads((frozen / "receipt.json").read_bytes()) == original_receipt
    path.write_bytes(path.read_bytes() + b" ")
    frozen_cases, frozen_receipt = FIT.validated_collection(frozen / "cases.jsonl", "train")
    assert frozen_cases[0].gold == {"operation": "DONE"}
    assert frozen_receipt == original_receipt
    verified = FIT._verify_collection(snapshot)
    assert verified["success"] is False
    assert "cases.jsonl" in verified["changed"]


def test_public_freeze_uses_validated_bytes_without_mutating_original_receipt(tmp_path):
    path = collection(tmp_path, "train")
    frozen = tmp_path / "public-frozen"
    original = (path.parent / "receipt.json").read_bytes()
    cases, receipt = FIT.validated_collection(path, "train", freeze_dir=frozen)
    assert cases[0].gold == {"operation": "DONE"}
    assert receipt == json.loads(original)
    assert (frozen / "cases.jsonl").read_bytes() == path.read_bytes()
    assert (frozen / "receipt.json").read_bytes() == original
    assert (path.parent / "receipt.json").read_bytes() == original


@pytest.mark.parametrize("change", ["receipt", "missing", "added"])
def test_final_input_inventory_detects_receipt_changes_and_file_set_changes(tmp_path, change):
    path = collection(tmp_path, "train")
    _cases, _receipt, snapshot = FIT._validated_snapshot(path, "train")
    expected = {
        "receipt": ("changed", "receipt.json"),
        "missing": ("missing", "fixtures/0000.html"),
        "added": ("added", "fixtures/0001.html"),
    }
    if change == "receipt":
        rewrite_receipt(path, lambda receipt: receipt.update(seed=12))
    elif change == "missing":
        (path.parent / "fixtures/0000.html").unlink()
    else:
        (path.parent / "fixtures/0001.html").write_text("Extra fixture")
    verified = FIT._verify_collection(snapshot)
    assert verified["success"] is False
    field, relative = expected[change]
    assert relative in verified[field]


def finalization(
    output, receipt, *, snapshots=None, source_manifest=None, primary_error=None, owned=()
):
    return FIT._finalize_fit(
        output,
        receipt,
        source_manifest=source_manifest,
        source_files={},
        input_snapshots={} if snapshots is None else snapshots,
        started=time.perf_counter(),
        primary_error=primary_error,
        owned_artifacts=owned,
    )


def test_final_receipt_drift_fails_run_and_invalidates_only_owned_sidecar(tmp_path):
    path = collection(tmp_path, "train")
    _cases, _receipt, snapshot = FIT._validated_snapshot(path, "train")
    output = tmp_path / "fit"
    output.mkdir()
    owned, unrelated = output / "learned", output / "unrelated"
    for folder in (owned, unrelated):
        folder.mkdir()
        (folder / "manifest.json").write_bytes(b'{"owned": true}\n')
        (folder / "weights.pt").write_bytes(b"preserved diagnostic bytes")
    rewrite_receipt(path, lambda receipt: receipt.update(stage="modified"))
    with pytest.raises(RuntimeError):
        finalization(output, {"stage": "complete"}, snapshots={"train": snapshot}, owned=(owned,))
    assert json.loads((output / "receipt.json").read_text())["stage"] == "failed"
    assert not (owned / "manifest.json").exists()
    assert (owned / "manifest.invalid.json").read_bytes() == b'{"owned": true}\n'
    assert (owned / "weights.pt").read_bytes() == b"preserved diagnostic bytes"
    assert (unrelated / "manifest.json").read_bytes() == b'{"owned": true}\n'


@pytest.mark.parametrize("failure", ["false", "exception"])
def test_final_source_verification_failure_records_failed_receipt_and_raises(
    tmp_path, monkeypatch, failure
):
    from s1 import browser_evidence

    output = tmp_path / "fit"
    output.mkdir()

    def verify(*_args, **_kwargs):
        if failure == "exception":
            raise OSError("source verification unavailable")
        return {"success": False, "changed": ["s1/browser_policy.py"], "missing": [], "added": []}

    monkeypatch.setattr(browser_evidence, "verify_runtime_sources", verify)
    with pytest.raises(RuntimeError):
        finalization(output, {"stage": "complete"}, source_manifest={"source": "captured"})
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["stage"] == "failed"
    assert "source" in json.dumps(receipt).lower()


def test_partial_owned_sidecar_is_invalidated_when_primary_failure_already_exists(tmp_path):
    output = tmp_path / "fit"
    output.mkdir()
    sidecar = output / "control"
    sidecar.mkdir()
    (sidecar / "manifest.json").write_bytes(b"partial manifest")
    (sidecar / "weights.pt").write_bytes(b"partial weights")
    primary = ValueError("original fitting failure")
    receipt = {"stage": "failed", "error": {"type": "ValueError", "message": str(primary)}}
    with pytest.raises(ValueError) as caught:
        try:
            raise primary
        except ValueError as exc:
            finalization(output, receipt, primary_error=exc, owned=(sidecar,))
            raise
    assert caught.value is primary
    saved = json.loads((output / "receipt.json").read_text())
    assert saved["stage"] == "failed"
    assert saved["error"] == receipt["error"]
    assert not (sidecar / "manifest.json").exists()
    assert (sidecar / "manifest.invalid.json").read_bytes() == b"partial manifest"
    assert (sidecar / "weights.pt").read_bytes() == b"partial weights"


def test_final_verifier_exception_does_not_replace_primary_training_exception(
    tmp_path, monkeypatch
):
    from s1 import browser_evidence

    output = tmp_path / "fit"
    output.mkdir()
    primary = ValueError("original training exception")

    def fail(*_args, **_kwargs):
        raise OSError("secondary source error")

    monkeypatch.setattr(browser_evidence, "verify_runtime_sources", fail)
    receipt = {"stage": "failed", "error": {"type": "ValueError", "message": str(primary)}}
    with pytest.raises(ValueError) as caught:
        try:
            raise primary
        except ValueError as exc:
            finalization(output, receipt, source_manifest={"source": "captured"}, primary_error=exc)
            raise
    assert caught.value is primary
    saved = json.loads((output / "receipt.json").read_text())
    assert saved["stage"] == "failed"
    assert saved["error"]["message"] == str(primary)
    assert "secondary source error" in json.dumps(saved)


def test_successful_finalization_keeps_valid_sidecar_and_complete_receipt(tmp_path, monkeypatch):
    from s1 import browser_evidence

    path = collection(tmp_path, "train")
    _cases, _receipt, snapshot = FIT._validated_snapshot(path, "train")
    output = tmp_path / "fit"
    output.mkdir()
    sidecar = output / "learned"
    sidecar.mkdir()
    manifest_bytes = b'{"valid": true}\n'
    (sidecar / "manifest.json").write_bytes(manifest_bytes)
    monkeypatch.setattr(
        browser_evidence,
        "verify_runtime_sources",
        lambda *_args, **_kwargs: {"success": True, "changed": [], "missing": [], "added": []},
    )
    finalization(
        output,
        {"stage": "complete"},
        snapshots={"train": snapshot},
        source_manifest={"source": "captured"},
        owned=(sidecar,),
    )
    assert json.loads((output / "receipt.json").read_text())["stage"] == "complete"
    assert (sidecar / "manifest.json").read_bytes() == manifest_bytes
    assert not (sidecar / "manifest.invalid.json").exists()


def add_prospective_sources(path):
    from s1.browser_evidence import capture_runtime_sources

    manifest = capture_runtime_sources(
        path.parent,
        additional_files={"scripts/jev_browser_training_data.py": Path(COLLECTOR.__file__)},
    )
    rewrite_receipt(
        path,
        lambda receipt: receipt.update(
            source_manifest=manifest,
            collector_sha256=manifest["files"]["scripts/jev_browser_training_data.py"]["sha256"],
            source_preservation_timing="before fixture generation and browser collection",
            source_verification={"success": True},
            upstream_verification={"success": True},
        ),
    )
    return manifest


@pytest.mark.parametrize("change", ["changed", "missing", "added", "upstream_failed"])
def test_prospective_source_bytes_and_final_verification_are_required(tmp_path, change):
    path = collection(tmp_path, "train")
    add_prospective_sources(path)
    FIT.validated_collection(path, "train")
    source = path.parent / "sources/scripts/jev_browser_training_data.py"
    if change == "changed":
        source.write_bytes(source.read_bytes() + b"\n# changed after collection\n")
    elif change == "missing":
        source.unlink()
    elif change == "added":
        (path.parent / "sources/s1/unrecorded.py").write_text("# extra source\n")
    else:
        rewrite_receipt(
            path, lambda receipt: receipt.update(upstream_verification={"success": False})
        )
    with pytest.raises(ValueError):
        FIT.validated_collection(path, "train")


def test_prospective_sources_are_frozen_and_original_changes_are_reverified(tmp_path):
    path = collection(tmp_path, "train")
    add_prospective_sources(path)
    _cases, _receipt, snapshot = FIT._validated_snapshot(path, "train")
    frozen = tmp_path / "prospective-frozen"
    FIT._freeze_collection(snapshot, frozen)
    source_relative = "sources/scripts/jev_browser_training_data.py"
    assert source_relative in snapshot["blobs"]
    assert (frozen / source_relative).read_bytes() == snapshot["blobs"][source_relative]
    FIT.validated_collection(frozen / "cases.jsonl", "train")
    original = path.parent / source_relative
    original.write_bytes(original.read_bytes() + b"\n# changed during fitting\n")
    verification = FIT._verify_collection(snapshot)
    assert verification["success"] is False
    assert source_relative in verification["changed"]
    assert (frozen / source_relative).read_bytes() == snapshot["blobs"][source_relative]


@pytest.mark.parametrize(
    "fault",
    ["none", "source", "original_receipt", "frozen_cases", "frozen_manifest", "partial_sidecar"],
)
@pytest.mark.parametrize("feature_batch_size", [1, 2])
def test_fit_lifecycle_freezes_before_model_and_final_integrity_controls_sidecars(
    tmp_path, monkeypatch, fault, feature_batch_size
):
    """Exercise fit wiring with model/feature/optimizer/serialization boundaries replaced."""
    from types import SimpleNamespace

    from s1 import browser_evidence, browser_policy, browser_readout, unified
    from s1.evaluation import datasets

    train = collection(tmp_path, "train", group="training episode")
    development = collection(tmp_path, "development", group="development episode")
    snapshots = {
        "train": FIT._validated_snapshot(train, "train")[2],
        "development": FIT._validated_snapshot(development, "development")[2],
    }
    output = tmp_path / "fit-run"
    audit_calls, model_calls, feature_calls, fit_calls = [], [], [], []
    actual_audit = datasets.audit_splits
    primary = OSError("sidecar write interrupted after manifest creation")

    class SelectedRows:
        def detach(self):
            return self

        def index_select(self, axis, letters):
            assert axis == 0 and letters == "original candidate IDs"
            return self

        def cpu(self):
            return self

        def clone(self):
            return SelectedRows()

    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(cuda=SimpleNamespace(enable_cudnn_sdp=lambda enabled: None)),
        cuda=SimpleNamespace(max_memory_allocated=lambda: 0),
        equal=lambda left, right: (
            isinstance(left, SelectedRows) and isinstance(right, SelectedRows)
        ),
        save=lambda _data, destination: Path(destination).write_bytes(
            b"model-free feature artifact"
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    def audit(paths):
        assert list(paths) == [
            output / "inputs/train/cases.jsonl",
            output / "inputs/development/cases.jsonl",
        ]
        audit_calls.append(list(paths))
        return actual_audit(paths)

    monkeypatch.setattr(datasets, "audit_splits", audit)

    def model(name, **options):
        assert len(audit_calls) == 1
        assert name == "knowlet/Gemma-4-12B-Unified-System-One"
        assert options["revision"] == "a66f836b56605039fe040f330180e336d19b3362"
        assert (output / "sources/manifest.json").is_file()
        for split, snapshot in snapshots.items():
            folder = output / "inputs" / split
            assert json.loads((folder / "input_manifest.json").read_text()) == snapshot["manifest"]
            for relative, blob in snapshot["blobs"].items():
                assert (folder / relative).read_bytes() == blob
        model_calls.append(name)
        if fault == "original_receipt":
            rewrite_receipt(train, lambda receipt: receipt.update(seed=99))
        if fault in {"frozen_cases", "frozen_manifest"}:
            filename = "cases.jsonl" if fault == "frozen_cases" else "input_manifest.json"
            frozen_file = output / "inputs/train" / filename
            frozen_file.write_bytes(frozen_file.read_bytes() + b" ")
        return SimpleNamespace(
            name=name,
            revision=options["revision"],
            head=SimpleNamespace(weight=SelectedRows()),
            letters="original candidate IDs",
            quantization_details={"mode": "nf4", "test_boundary": True},
            temperature=2.7830344470383452,
        )

    monkeypatch.setattr(unified, "UnifiedDecisionModel", model)
    monkeypatch.setattr(
        browser_policy,
        "BrowserDecisionBackend",
        lambda loaded, **_kwargs: SimpleNamespace(
            model=loaded, prompt_variant="grounded_user_question_v1"
        ),
    )

    def features(_backend, cases, *, feature_batch_size: int, evidence):
        assert feature_batch_size == requested_batch_size
        feature_calls.append([case.id for case in cases])
        evidence.update(status="completed", forward_calls=1, model_free_lifecycle_test=True)
        return [{"case_id": case.id, "split": case.split} for case in cases]

    monkeypatch.setattr(FIT, "feature_rows", features)
    requested_batch_size = feature_batch_size
    batch_options = {"feature_batch_size": feature_batch_size} if feature_batch_size == 2 else {}

    def fit_head(_model, rows, **_kwargs):
        assert all(row["split"] == "train" for row in rows)
        fit_calls.append(rows)
        return {"weight": "selected learned rows", "bias": "selected learned bias"}, {"steps": 1}

    # The source capture resolves this callable's filename; keep that boundary
    # pinned to the actual fitting helper while replacing the CPU optimizer.
    fit_head.__code__ = fit_head.__code__.replace(co_filename=FIT.fit_head.__code__.co_filename)
    monkeypatch.setattr(FIT, "fit_head", fit_head)

    def write_readout(destination, _model, _variant, **_kwargs):
        destination.mkdir()
        (destination / "weights.pt").write_bytes(b"model-free selected rows")
        (destination / "manifest.json").write_bytes(b'{"test_sidecar": true}\n')
        if fault == "partial_sidecar" and destination.name == "learned":
            raise primary
        return {"test_sidecar": True}

    monkeypatch.setattr(browser_readout, "write_readout", write_readout)
    monkeypatch.setattr(
        browser_evidence,
        "verify_runtime_sources",
        lambda *_args, **_kwargs: {
            "success": fault != "source",
            "changed": [],
            "missing": [],
            "added": [],
        },
    )
    if fault == "none":
        result = FIT.fit(
            train_cases=train,
            development_cases=development,
            output_dir=output,
            steps=1,
            **batch_options,
        )
        assert result["stage"] == "complete"
    else:
        error_type = OSError if fault == "partial_sidecar" else RuntimeError
        with pytest.raises(error_type) as caught:
            FIT.fit(
                train_cases=train,
                development_cases=development,
                output_dir=output,
                steps=1,
                **batch_options,
            )
        if fault == "partial_sidecar":
            assert caught.value is primary
    assert len(model_calls) == 1 and len(feature_calls) == 2 and len(fit_calls) == 1
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["feature_batch_size"] == feature_batch_size
    assert receipt["backbone_forward_calls_this_run"] == 2
    assert receipt["feature_capture"]["feature_batch_size"] == feature_batch_size
    assert receipt["feature_capture_sha256"] == digest(
        (output / "feature_capture.json").read_bytes()
    )
    assert receipt["split_audit"]["valid"] is True
    assert receipt["split_audit_paths"] == [
        "inputs/train/cases.jsonl",
        "inputs/development/cases.jsonl",
    ]
    assert receipt["stage"] == ("complete" if fault == "none" else "failed")
    for name in ("control", "learned"):
        artifact = output / name
        assert (artifact / "weights.pt").read_bytes() == b"model-free selected rows"
        assert (artifact / "manifest.json").exists() is (fault == "none")
        assert (artifact / "manifest.invalid.json").exists() is (fault != "none")
    if fault == "original_receipt":
        assert receipt["input_verification"]["train"]["changed"] == ["receipt.json"]
        assert (output / "inputs/train/receipt.json").read_bytes() == snapshots["train"]["blobs"][
            "receipt.json"
        ]

    if fault == "frozen_cases":
        assert receipt["frozen_input_verification"]["train"]["changed"] == ["cases.jsonl"]
        assert receipt["input_verification"]["train"]["success"] is True
    if fault == "frozen_manifest":
        assert receipt["frozen_input_verification"]["train"]["manifest_changed"] is True
        assert receipt["input_verification"]["train"]["success"] is True


@pytest.mark.parametrize("creation", ["existing", "race"])
def test_unowned_existing_or_racing_sidecar_is_never_invalidated(tmp_path, creation):
    output = tmp_path / "fit"
    output.mkdir()
    sidecar = output / "control"
    owned, called = [], []
    manifest_bytes = b'{"owned_by_other_caller": true}\n'
    if creation == "existing":
        sidecar.mkdir()
        (sidecar / "manifest.json").write_bytes(manifest_bytes)

    def competing_writer(destination):
        called.append(destination)
        destination.mkdir()
        (destination / "manifest.json").write_bytes(manifest_bytes)
        raise FileExistsError("competing creator owns this destination")

    with pytest.raises(FileExistsError) as caught:
        FIT._write_owned_readout(competing_writer, sidecar, owned)
    assert owned == []
    assert len(called) == int(creation == "race")
    receipt = {
        "stage": "failed",
        "error": {"type": "FileExistsError", "message": str(caught.value)},
    }
    finalization(output, receipt, primary_error=caught.value, owned=owned)
    assert (sidecar / "manifest.json").read_bytes() == manifest_bytes
    assert not (sidecar / "manifest.invalid.json").exists()


def test_partial_sidecar_writer_retains_ownership_for_failure_cleanup(tmp_path):
    output = tmp_path / "fit"
    output.mkdir()
    sidecar, owned = output / "learned", []
    primary = OSError("partial write failure")

    def interrupted_writer(destination):
        destination.mkdir()
        (destination / "manifest.json").write_bytes(b"partial owned manifest")
        raise primary

    with pytest.raises(OSError) as caught:
        FIT._write_owned_readout(interrupted_writer, sidecar, owned)
    assert caught.value is primary and owned == [sidecar]
    receipt = {"stage": "failed", "error": {"type": "OSError", "message": str(primary)}}
    finalization(output, receipt, primary_error=primary, owned=owned)
    assert not (sidecar / "manifest.json").exists()
    assert (sidecar / "manifest.invalid.json").read_bytes() == b"partial owned manifest"


def test_receipt_write_failure_adds_note_without_replacing_primary_exception(tmp_path):
    output = tmp_path / "fit"
    output.mkdir()
    (output / "receipt.json").mkdir()
    primary = ValueError("original training failure")
    receipt = {"stage": "failed", "error": {"type": "ValueError", "message": str(primary)}}
    with pytest.raises(ValueError) as caught:
        try:
            raise primary
        except ValueError as exc:
            finalization(output, receipt, primary_error=exc)
            raise
    assert caught.value is primary
    assert any("final training receipt" in note for note in primary.__notes__)


@pytest.mark.parametrize("feature_batch_size", [None, True, False, 0, 3, 1.0, "2"])
def test_feature_batch_option_is_strict_before_input_reads_or_output(
    tmp_path, monkeypatch, feature_batch_size
):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid batch option reached collection validation")

    monkeypatch.setattr(FIT, "_validated_snapshot", forbidden)
    output = tmp_path / "invalid-fit"
    with pytest.raises(ValueError, match="integer 1 or 2"):
        FIT.fit(
            train_cases="unread-train",
            development_cases="unread-development",
            output_dir=output,
            feature_batch_size=feature_batch_size,
        )
    assert not output.exists()


def test_capture_batch_option_is_rejected_for_cpu_replay_before_output(tmp_path, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("inapplicable batch option reached collection validation")

    monkeypatch.setattr(FIT, "_validated_snapshot", forbidden)
    output = tmp_path / "invalid-replay"
    with pytest.raises(ValueError, match="does not apply to CPU feature replay"):
        FIT.fit(
            train_cases="unread-train",
            development_cases="unread-development",
            output_dir=output,
            features_from="unread-origin",
            feature_batch_size=2,
        )
    assert not output.exists()


@pytest.mark.parametrize("option,expected", [([], 1), (["--feature-batch-size", "2"], 2)])
def test_feature_batch_cli_keeps_default_and_accepts_opt_in(monkeypatch, capsys, option, expected):
    received = []

    def fit_stub(**kwargs):
        received.append(kwargs)
        return dict(stage="complete", scope="CLI test", learned_artifact="l", control_artifact="c")

    monkeypatch.setattr(FIT, "fit", fit_stub)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jev_browser_fit_readout.py",
            "--train-cases",
            "train.jsonl",
            "--development-cases",
            "development.jsonl",
            "--output-dir",
            "fit-output",
            *option,
        ],
    )
    FIT.main()
    assert len(received) == 1 and received[0]["feature_batch_size"] == expected
    assert json.loads(capsys.readouterr().out)["stage"] == "complete"


def test_feature_batch_cli_rejects_unsupported_size_before_fit(monkeypatch):
    def forbidden(**_kwargs):
        raise AssertionError("invalid CLI batch option reached fit")

    monkeypatch.setattr(FIT, "fit", forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "jev_browser_fit_readout.py",
            "--train-cases",
            "train.jsonl",
            "--development-cases",
            "development.jsonl",
            "--output-dir",
            "fit-output",
            "--feature-batch-size",
            "3",
        ],
    )
    with pytest.raises(SystemExit) as error:
        FIT.main()
    assert error.value.code == 2
