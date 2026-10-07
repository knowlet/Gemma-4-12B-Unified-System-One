"""Publication gates and immutable verification run without GPU or Hub writes."""

import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def publisher():
    return runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/publish_nvfp4.py"))


@pytest.fixture
def package(tmp_path, monkeypatch, publisher):
    root = tmp_path / "package"
    root.mkdir()
    for name in publisher["REQUIRED_FILES"]:
        (root / name).write_text("{}\n")
    (root / "README.md").write_text("# Packed NVFP4 release\n")
    (root / "model.safetensors").write_bytes(b"packed fixture checked by core validator")
    (root / "nvfp4_manifest.json").write_text(
        json.dumps({"tensors": {"linear.weight": {"filename": "model.safetensors"}}})
    )
    files = {
        name: {
            "sha256": publisher["file_sha256"](root / name),
            "size_bytes": (root / name).stat().st_size,
        }
        for name in (
            "config.json",
            "s1_config.json",
            "nvfp4_manifest.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }
    conversion = {
        "format": "nvfp4",
        "status": "ok",
        "structural_verification": {"status": "ok"},
        "files": files,
    }
    (root / "conversion-manifest.json").write_text(json.dumps(conversion))
    evaluation = {
        "status": "ok",
        "release_validation_complete": True,
        "population_complete": True,
        "conversion_manifest_sha256": publisher["file_sha256"](root / "conversion-manifest.json"),
        "reports": {
            name: {"coverage": 1.0, "counts": {"ok": count}, "metrics": {"n": count}}
            for name, count in publisher["REQUIRED_POPULATIONS"].items()
        },
    }
    (root / "evaluation.json").write_text(json.dumps(evaluation))
    wheel = "gemma_system_one-0.1.0-py3-none-any.whl"
    (root / wheel).write_bytes(b"exact tested runtime wheel")
    runtime_verification = {
        "status": "ok",
        "population_complete": True,
        "conversion_manifest_sha256": publisher["file_sha256"](root / "conversion-manifest.json"),
        "evaluation_sha256": publisher["file_sha256"](root / "evaluation.json"),
        "runtime_wheel": {"filename": wheel, "sha256": publisher["file_sha256"](root / wheel)},
        "reports": evaluation["reports"],
    }
    (root / "runtime-verification.json").write_text(json.dumps(runtime_verification))
    checked = []

    def validate(path):
        checked.append(path)
        return {"status": "ok", "format": "nvfp4", "quantized_linear_modules": 1}

    monkeypatch.setitem(sys.modules, "s1.nvfp4", SimpleNamespace(validate_nvfp4_artifact=validate))
    names = sorted(path.name for path in root.iterdir())
    return SimpleNamespace(
        root=root,
        names=names,
        conversion=conversion,
        evaluation=evaluation,
        runtime_verification=runtime_verification,
        checked=checked,
    )


def rewrite_evaluation(package):
    (package.root / "evaluation.json").write_text(json.dumps(package.evaluation))


def test_validation_hashes_only_exact_whitelist_and_checks_packed_artifact(publisher, package):
    (package.root / "unrelated-private-file.txt").write_text("not in release")
    result = publisher["validate_package"](package.root, package.names)
    assert result["status"] == "validated_locally"
    assert {entry["path"] for entry in result["files"]} == set(package.names)
    assert package.checked == [package.root]
    assert result["preupload_inventory_sha256"] == publisher["digest"](result["files"])


@pytest.mark.parametrize(
    "name", ["../file", "/tmp/file", "sub/../file", "*.json", "sub\\file", "a//b"]
)
def test_exact_whitelist_rejects_traversal_and_patterns(publisher, package, name):
    with pytest.raises(ValueError, match="path"):
        publisher["validate_package"](package.root, [*package.names, name])


def test_symlink_cannot_publish_outside_file(publisher, package, tmp_path):
    outside = tmp_path / "private.txt"
    outside.write_text("not a release file")
    (package.root / "linked.txt").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic links"):
        publisher["validate_package"](package.root, [*package.names, "linked.txt"])


def test_conversion_hash_binds_actual_weight_bytes(publisher, package):
    (package.root / "model.safetensors").write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="identity changed"):
        publisher["validate_package"](package.root, package.names)


def test_omitted_checkpoint_shard_is_rejected(publisher, package):
    with pytest.raises(ValueError, match="omits converted"):
        publisher["validate_package"](
            package.root, [name for name in package.names if name != "model.safetensors"]
        )


def test_conversion_cannot_claim_success_without_core_packed_validation(
    publisher, package, monkeypatch
):
    def reject(path):
        raise ValueError("dense weights are not a packed checkpoint")

    monkeypatch.setitem(sys.modules, "s1.nvfp4", SimpleNamespace(validate_nvfp4_artifact=reject))
    with pytest.raises(ValueError, match="not a packed checkpoint"):
        publisher["validate_package"](package.root, package.names)


@pytest.mark.parametrize(
    "failure", ["manifest", "incomplete", "small_population", "errors", "metrics"]
)
def test_evaluation_must_bind_full_successful_population(publisher, package, failure):
    report = package.evaluation["reports"]["test"]
    if failure == "manifest":
        package.evaluation["conversion_manifest_sha256"] = "f" * 64
    elif failure == "incomplete":
        package.evaluation["population_complete"] = False
    elif failure == "small_population":
        report["counts"]["ok"] = 2
    elif failure == "errors":
        report["counts"]["error"] = 1
    else:
        report["metrics"]["n"] = 1
    rewrite_evaluation(package)
    with pytest.raises(ValueError, match="evaluation"):
        publisher["validate_package"](package.root, package.names)


def test_validate_only_never_initializes_hub(publisher, package, tmp_path, monkeypatch):
    def forbidden():
        pytest.fail("validate-only must not access Hub")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=forbidden))
    path = tmp_path / "receipt.json"
    result = publisher["publish_package"](
        package.root, package.names, publisher["DEFAULT_REPOSITORY"], path, validate_only=True
    )
    assert result["status"] == json.loads(path.read_text())["status"] == "validated_locally"
    with pytest.raises(ValueError, match="outside"):
        publisher["publish_package"](
            package.root,
            package.names,
            publisher["DEFAULT_REPOSITORY"],
            package.root / "receipt.json",
            validate_only=True,
        )


@pytest.fixture
def fake_hub(publisher, package, monkeypatch):
    records = []
    revision = "b" * 40
    files = publisher["validate_package"](package.root, package.names)["files"]
    state = SimpleNamespace(
        records=records, revision=revision, files=files, corrupt=None, extra=False, private=False
    )

    class API:
        def __init__(self, token=None):
            self.token = token

        def create_repo(self, **kwargs):
            records.append(("create_repo", kwargs))

        def model_info(self, repo_id, **kwargs):
            records.append(("model_info", kwargs))
            immutable = kwargs.get("revision")
            siblings = [SimpleNamespace(rfilename=".gitattributes")]
            if immutable:
                assert immutable == revision and kwargs["token"] is False
                for entry in files:
                    sha = "f" * 64 if state.corrupt == entry["path"] else entry["sha256"]
                    siblings.append(
                        SimpleNamespace(
                            rfilename=entry["path"],
                            size=entry["size_bytes"],
                            lfs=(
                                {"sha256": sha, "size": entry["size_bytes"]}
                                if entry["path"].endswith("safetensors")
                                else None
                            ),
                        )
                    )
            if state.extra:
                siblings.append(SimpleNamespace(rfilename="unlisted-file.json"))
            return SimpleNamespace(
                sha=revision if immutable else "a" * 40,
                private=state.private,
                siblings=siblings,
            )

        def create_commit(self, **kwargs):
            records.append(("create_commit", kwargs))
            return SimpleNamespace(
                oid=revision, commit_url=f"https://huggingface.co/commit/{revision}"
            )

    def download(**kwargs):
        records.append(("download", kwargs))
        assert kwargs["revision"] == revision
        assert kwargs["force_download"] is True and kwargs["token"] is False
        name = kwargs["filename"]
        if state.corrupt == name:
            (package.root / name).write_bytes(b"bad bytes")
        return str(package.root / name)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            HfApi=API,
            CommitOperationAdd=lambda **kwargs: SimpleNamespace(**kwargs),
            get_token=lambda: "test-authentication",
            hf_hub_download=download,
        ),
    )
    return state


def test_publication_uses_exact_operations_and_verifies_immutable_files(
    publisher, package, tmp_path, fake_hub
):
    path = tmp_path / "receipt.json"
    result = publisher["publish_package"](
        package.root, package.names, publisher["DEFAULT_REPOSITORY"], path
    )
    assert result["status"] == "verified_public"
    assert json.loads(path.read_text())["published_revision"] == fake_hub.revision
    commit = next(args for name, args in fake_hub.records if name == "create_commit")
    assert commit["parent_commit"] == "a" * 40
    assert {operation.path_in_repo for operation in commit["operations"]} == set(package.names)
    assert all(
        operation.path_or_fileobj.parent == package.root for operation in commit["operations"]
    )
    weights = next(row for row in result["files"] if row["path"] == "model.safetensors")
    assert weights["verification"] == "remote_lfs_sha256_and_size"
    assert all(
        row["verification"] == "anonymous_immutable_download_sha256_and_size"
        for row in result["files"]
        if row["path"] != "model.safetensors"
    )


@pytest.mark.parametrize("name", ["model.safetensors", "README.md"])
def test_uploaded_revision_failure_never_claims_verified(
    publisher, package, tmp_path, fake_hub, name
):
    fake_hub.corrupt = name
    path = tmp_path / "receipt.json"
    with pytest.raises(ValueError, match="differs"):
        publisher["publish_package"](
            package.root, package.names, publisher["DEFAULT_REPOSITORY"], path
        )
    saved = json.loads(path.read_text())
    assert saved["status"] == "verification_failed"
    assert saved["published_revision"] == fake_hub.revision
    assert "verified_at" not in saved


def test_existing_unlisted_hub_file_stops_before_upload(publisher, package, tmp_path, fake_hub):
    fake_hub.extra = True
    with pytest.raises(ValueError, match="unlisted files"):
        publisher["publish_package"](
            package.root, package.names, publisher["DEFAULT_REPOSITORY"], tmp_path / "receipt.json"
        )
    assert not any(name == "create_commit" for name, _ in fake_hub.records)


def test_public_verification_rejects_mutable_revision(publisher):
    with pytest.raises(ValueError, match="immutable"):
        publisher["verify_remote"](None, None, "owner/model", "main", [], None)


def test_inventory_rejects_malformed_hash(publisher):
    with pytest.raises(ValueError, match="identity"):
        publisher["inventory_entries"](
            {"model.safetensors": {"size_bytes": 100, "sha256": hashlib.sha1(b"wrong").hexdigest()}}
        )


@pytest.mark.parametrize("failure", ["failed", "partial", "wheel", "evaluation", "conversion"])
def test_runtime_verification_binds_published_wheel_and_full_artifact(publisher, package, failure):
    verification = package.runtime_verification
    if failure == "failed":
        verification["status"] = "failed"
    elif failure == "partial":
        verification["reports"]["media"]["counts"]["ok"] = 1
    elif failure == "wheel":
        (package.root / verification["runtime_wheel"]["filename"]).write_bytes(b"untested runtime")
    elif failure == "evaluation":
        verification["evaluation_sha256"] = "f" * 64
    else:
        verification["conversion_manifest_sha256"] = "f" * 64
    (package.root / "runtime-verification.json").write_text(json.dumps(verification))
    with pytest.raises(ValueError, match="runtime"):
        publisher["validate_package"](package.root, package.names)


def test_verified_runtime_wheel_cannot_be_omitted_from_upload(publisher, package):
    filename = package.runtime_verification["runtime_wheel"]["filename"]
    with pytest.raises(ValueError, match="runtime wheel"):
        publisher["validate_package"](
            package.root, [name for name in package.names if name != filename]
        )
