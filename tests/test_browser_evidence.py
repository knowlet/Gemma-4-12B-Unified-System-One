"""Actual source identity is separate from model identity and copied byte-for-byte."""

import hashlib
import json

import pytest

from s1 import browser_evidence


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    files = {
        "src/s1/__init__.py": b'"""fixture package"""\r\n',
        "src/s1/browser_evidence.py": b"# source evidence fixture\n",
        "src/s1/browser_policy.py": b'PROMPT = "old adapter"\n',
        "src/s1/evaluation/jev_omni.py": b'REVISION = "pinned"\n',
        "scripts/jev_browser_benchmark.py": b"# exact browser runner\r\n",
        "scripts/jev_local_model.py": b"# local caller\n",
        "scripts/jev_public_web_cases.py": b"# external public-site judge\n",
    }
    for relative, blob in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    # These deliberately out-of-scope files must never reach public evidence.
    (root / ".env").write_text("API_KEY=do-not-capture")
    (root / "scripts" / "private.py").write_text("credential = 'do-not-capture'")
    monkeypatch.setattr(browser_evidence, "__file__", str(root / "src/s1/browser_evidence.py"))
    return root, files


def test_capture_uses_imported_package_and_preserves_original_bytes(source_tree, tmp_path):
    root, originals = source_tree
    output = tmp_path / "evidence"
    manifest = browser_evidence.capture_runtime_sources(output)
    expected = {relative.removeprefix("src/") for relative in originals}
    assert set(manifest["files"]) == expected
    assert json.loads((output / "sources/manifest.json").read_text()) == manifest
    assert str(root) not in json.dumps(manifest)
    for relative, blob in originals.items():
        label = relative.removeprefix("src/")
        record = manifest["files"][label]
        assert (output / record["artifact_path"]).read_bytes() == blob
        assert record["sha256"] == hashlib.sha256(blob).hexdigest()
        assert record["size_bytes"] == len(blob)
    assert browser_evidence.verify_runtime_sources(manifest)["success"] is True
    assert "do-not-capture" not in "".join(
        path.read_text() for path in (output / "sources").rglob("*.py")
    )


def test_changed_adapter_invalidates_source_identity_despite_same_model_pin(source_tree, tmp_path):
    root, _originals = source_tree
    metadata = {"model": "same-checkpoint", "revision": "a" * 40, "prompt_mode": "user"}
    original = browser_evidence.capture_runtime_sources(tmp_path / "first")
    (root / "src/s1/browser_policy.py").write_text('PROMPT = "changed adapter"\n')
    check = browser_evidence.verify_runtime_sources(original)
    assert check["success"] is False
    assert check["changed"] == ["s1/browser_policy.py"]
    changed = browser_evidence.capture_runtime_sources(tmp_path / "second")
    assert {"model_metadata": metadata, "source_identity": original["identity_sha256"]} != {
        "model_metadata": metadata,
        "source_identity": changed["identity_sha256"],
    }
    assert check["current_identity_sha256"] == changed["identity_sha256"]


def test_added_and_removed_package_sources_are_detected(source_tree, tmp_path):
    root, _originals = source_tree
    manifest = browser_evidence.capture_runtime_sources(tmp_path / "evidence")
    (root / "src/s1/browser_policy.py").unlink()
    (root / "src/s1/new_adapter.py").write_text("# newly imported adapter")
    check = browser_evidence.verify_runtime_sources(manifest)
    assert check["success"] is False
    assert check["missing"] == ["s1/browser_policy.py"]
    assert check["added"] == ["s1/new_adapter.py"]


def test_named_cloud_caller_can_be_captured_from_actual_runtime_path(source_tree, tmp_path):
    _root, _originals = source_tree
    caller = tmp_path / "actual-runtime/jev_browser.py"
    caller.parent.mkdir()
    caller.write_bytes(b"# serialized cloud caller\n")
    supplied = {"apps/modal/jev_browser.py": caller}
    manifest = browser_evidence.capture_runtime_sources(
        tmp_path / "evidence", additional_files=supplied
    )
    assert "apps/modal/jev_browser.py" in manifest["files"]
    assert browser_evidence.verify_runtime_sources(manifest, additional_files=supplied)["success"]
    caller.write_text("# changed caller")
    check = browser_evidence.verify_runtime_sources(manifest, additional_files=supplied)
    assert check["changed"] == ["apps/modal/jev_browser.py"]
    assert str(caller) not in json.dumps(manifest)


def test_manifest_mutation_cannot_mask_changed_sources(source_tree, tmp_path):
    root, _originals = source_tree
    manifest = browser_evidence.capture_runtime_sources(tmp_path / "evidence")
    changed = b"# changed source\n"
    (root / "src/s1/browser_policy.py").write_bytes(changed)
    record = manifest["files"]["s1/browser_policy.py"]
    record.update(sha256=hashlib.sha256(changed).hexdigest(), size_bytes=len(changed))
    check = browser_evidence.verify_runtime_sources(manifest)
    assert check["success"] is False
    assert "identity was modified" in check["error"]


def test_existing_evidence_is_never_overwritten(source_tree, tmp_path):
    _root, _originals = source_tree
    destination = tmp_path / "evidence"
    original = browser_evidence.capture_runtime_sources(destination)
    with pytest.raises(FileExistsError):
        browser_evidence.capture_runtime_sources(destination)
    assert json.loads((destination / "sources/manifest.json").read_text()) == original


@pytest.mark.parametrize("label", ["../../private.py", "scripts/private.py", ".env", "s1/new.py"])
def test_unnamed_additional_sources_are_rejected(source_tree, tmp_path, label):
    root, _originals = source_tree
    with pytest.raises(ValueError, match="approved benchmark caller"):
        browser_evidence.capture_runtime_sources(
            tmp_path / "evidence", additional_files={label: root / ".env"}
        )


def test_missing_explicit_caller_is_detected_after_capture(source_tree, tmp_path):
    _root, _originals = source_tree
    caller = tmp_path / "jev_browser.py"
    caller.write_text("# actual caller")
    supplied = {"apps/modal/jev_browser.py": caller}
    manifest = browser_evidence.capture_runtime_sources(
        tmp_path / "evidence", additional_files=supplied
    )
    caller.unlink()
    check = browser_evidence.verify_runtime_sources(manifest, additional_files=supplied)
    assert check["success"] is False
    assert check["missing"] == ["apps/modal/jev_browser.py"]


def test_action_trainer_is_an_explicitly_named_source(source_tree, tmp_path):
    _root, _files = source_tree
    caller = tmp_path / "jev_browser_fit_actions.py"
    caller.write_bytes(b"# exact action trainer\r\n")
    supplied = {"scripts/jev_browser_fit_actions.py": caller}
    manifest = browser_evidence.capture_runtime_sources(
        tmp_path / "evidence", additional_files=supplied
    )
    assert browser_evidence.verify_runtime_sources(manifest, additional_files=supplied)["success"]
    caller.write_text("# modified action trainer")
    assert browser_evidence.verify_runtime_sources(manifest, additional_files=supplied)[
        "changed"
    ] == ["scripts/jev_browser_fit_actions.py"]
