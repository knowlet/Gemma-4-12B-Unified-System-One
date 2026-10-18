"""Pure public-weight source/profile/header tests; no SDK/network/framework."""

import ast
import copy
import hashlib
import importlib.util
import json
import struct
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "apps/modal/decision3_public_weights.py"


def prefix():
    module = types.ModuleType("pure_decision3_public_weights")
    module.__file__ = str(PATH)
    tree = ast.parse(PATH.read_bytes())
    stop = next(
        index
        for index, node in enumerate(tree.body)
        if isinstance(node, ast.Import) and any(alias.name == "modal" for alias in node.names)
    )
    exec(
        compile(ast.Module(body=tree.body[:stop], type_ignores=[]), str(PATH), "exec"),
        module.__dict__,
    )
    return module


WEIGHTS = prefix()
SERVICE = WEIGHTS.load_service()
spec = importlib.util.spec_from_file_location("pure_public_weights_core", SERVICE.CORE)
CORE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CORE)


def tiny_tensor(path, header, data):
    blob = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + data)


def cache(tmp_path, *, single_file=False):
    backbone = "model.safetensors" if single_file else "model-00001-of-00001.safetensors"
    readout = "readout.safetensors"
    tiny_tensor(
        tmp_path / backbone,
        {"weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}},
        b"\0" * 4,
    )
    tiny_tensor(
        tmp_path / readout,
        {"weight": {"dtype": "F32", "shape": [255, 2], "data_offsets": [0, 2040]}},
        b"\0" * 2040,
    )
    if not single_file:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": backbone}})
        )
    (tmp_path / "config.json").write_text('{"fixture":"no model"}')
    files = {p.name: CORE.pin(p) for p in tmp_path.iterdir()}
    manifest = {
        "schema": "d3-package-manifest/1",
        "model_name": "d3",
        "repo_id": "public-fixture/d3",
        "files_sha256": {name: pin["sha256"] for name, pin in files.items()},
        "files_bytes": {name: pin["size_bytes"] for name, pin in files.items()},
    }
    identity = {
        "model_id": "public-fixture/d3",
        "revision": "f" * 40,
        "manifest_sha256": "a" * 64,
        "readout_tensor_bytes": 2040,
        "loaded_parameters": 512,
        "resident_file_bytes": files[backbone]["size_bytes"] + files[readout]["size_bytes"],
    }
    staging = SimpleNamespace(
        provider_cache_file=lambda _binding, _identity, label: tmp_path / label,
        runtime_files=lambda value: sorted(value["files_sha256"]),
    )
    return staging, identity, manifest, files


def test_source18_is_service17_plus_only_downloader_and_preparation_is_stdlib(tmp_path):
    before = set(sys.modules)
    actual = WEIGHTS.inventory()
    original = SERVICE.source_inventory()
    assert actual["files"] == len(actual["rows"]) == 18
    assert {row["path"] for row in actual["rows"]} - {row["path"] for row in original["rows"]} == {
        str(PATH)
    }
    assert (
        actual["service_source_inventory_sha256"]
        == hashlib.sha256(SERVICE.encode(original)).hexdigest()
    )
    assert not actual["collection_or_parent_uploaded"] and not actual["weights_uploaded"]
    WEIGHTS.prepare(tmp_path / "prepared")
    assert json.loads((tmp_path / "prepared/plan.json").read_bytes())["GPU_requested"] is False
    assert not ({"torch", "modal"} & (set(sys.modules) - before))
    with pytest.raises(ValueError, match="Fresh"):
        WEIGHTS.prepare(tmp_path / "prepared")


@pytest.mark.parametrize("single_file", [False, True])
@pytest.mark.parametrize(
    "fault",
    [None, "readout_shape", "missing_shard", "parameters", "empty_backbone", "backbone_dtype"],
)
def test_real_stdlib_tensor_header_and_all_shard_geometry(tmp_path, fault, single_file):
    staging, identity, manifest, files = cache(tmp_path, single_file=single_file)
    backbone = next(
        name for name in files if name.endswith(".safetensors") and name != "readout.safetensors"
    )
    if fault == "readout_shape":
        tiny_tensor(
            tmp_path / "readout.safetensors",
            {"weight": {"dtype": "F32", "shape": [2, 255], "data_offsets": [0, 2040]}},
            b"\0" * 2040,
        )
    elif fault == "missing_shard":
        files.pop(backbone)
    elif fault == "parameters":
        identity["loaded_parameters"] += 1
    elif fault == "empty_backbone":
        tiny_tensor(tmp_path / backbone, {}, b"")
    elif fault == "backbone_dtype":
        tiny_tensor(
            tmp_path / backbone,
            {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
            b"\0" * 8,
        )
    if fault:
        with pytest.raises(RuntimeError):
            WEIGHTS.verify_geometry(CORE, staging, {}, identity, manifest, files)
    else:
        result = WEIGHTS.verify_geometry(CORE, staging, {}, identity, manifest, files)
        assert result["readout_shape"] == [255, 2] and result["full_payload_sha256_authenticated"]
        assert result["backbone_layout"] == ("single_file" if single_file else "indexed_shards")
        assert result["index_file"] == (None if single_file else "model.safetensors.index.json")
        assert (tmp_path / "model.safetensors.index.json").exists() is not single_file


@pytest.mark.parametrize("corrupt", [False, True])
def test_download_requests_only_exact_public_files_and_full_sha(tmp_path, monkeypatch, corrupt):
    staging, identity, manifest, files = cache(tmp_path)
    calls = []
    output = tmp_path / "output"
    output.mkdir()
    staging.mount_root = lambda _binding: tmp_path
    staging.provider_snapshot = lambda _binding, _identity: tmp_path
    identities = {
        name: {**identity, "model_id": "public-fixture/" + name} for name in SERVICE.MODELS
    }
    fake_core = SimpleNamespace(
        pin=CORE.pin,
        backbone_layout=CORE.backbone_layout,
        tensor_header=CORE.tensor_header,
        require=CORE.require,
        resource_snapshot=lambda: {"synthetic": "CPU-only"},
        authenticate_source=lambda _sha: {},
        write=lambda path, value: SERVICE.save(path, value),
    )

    def read(path, expected=None):
        if path.name == "MODEL_MANIFEST.json":
            name = calls[-1]["repo_id"].split("/")[-1]
            return {**manifest, "model_name": name, "repo_id": "public-fixture/" + name}
        return CORE.read(path, expected)

    fake_core.read = read

    def snapshot_download(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        if corrupt and len(calls) == 2:
            with (tmp_path / "config.json").open("ab") as stream:
                stream.write(b" ")
        return str(tmp_path)

    monkeypatch.setattr(WEIGHTS, "authenticate_manifest", lambda _blob: None)
    if corrupt:
        with pytest.raises(ValueError, match="SHA/size"):
            WEIGHTS.download_models(
                SERVICE, staging, fake_core, identities, {}, output, snapshot_download
            )
        assert not (output / "download-result.json").exists()
    else:
        result = WEIGHTS.download_models(
            SERVICE, staging, fake_core, identities, {}, output, snapshot_download
        )
        assert list(result["models"]) == list(SERVICE.MODELS)
        assert (
            result["token"] is False
            and result["model_loads"] == 0
            and result["GPU_requested"] is False
        )
        assert len(calls) == 12
        assert all(
            call["token"] is False
            and call["revision"] == identity["revision"]
            and call["max_workers"] == 1
            for call in calls
        )
        assert calls[0]["allow_patterns"] == ["MODEL_MANIFEST.json"]
        assert set(calls[1]["allow_patterns"]) == {"MODEL_MANIFEST.json", *files}


@pytest.mark.parametrize("complete", [True, False])
def test_profile_preserves_original_closed_export_bytes_and_refuses_failed_cpu(tmp_path, complete):
    identities = {
        name: {"revision": "f" * 40, "manifest_sha256": "a" * 64} for name in SERVICE.MODELS
    }
    controller = {"owned_exit_confirmed": True, "child_success": complete}
    result_json = {
        "token": False,
        "model_loads": 0,
        "GPU_requested": False,
        "models": {
            name: {**identity, "files": {"actual": {"sha256": "b" * 64, "size_bytes": 1}}}
            for name, identity in identities.items()
        },
    }
    originals = {
        "controller-receipt.json": json.dumps(controller, indent=2).encode(),
        "download-result.json": json.dumps(result_json, indent=1).encode(),
    }
    for label, raw in originals.items():
        (tmp_path / label).write_bytes(raw)
    (tmp_path / "export-index.json").write_bytes(
        SERVICE.encode(
            {
                "root": "fixture",
                "complete_owned_exit": True,
                "files": {label: SERVICE.pin(tmp_path / label) for label in originals},
            }
        )
    )
    rpc = {
        "success": complete,
        "owned_exit_confirmed": True,
        "export_index": SERVICE.pin(tmp_path / "export-index.json"),
    }
    if not complete:
        with pytest.raises(ValueError, match="successful closed"):
            WEIGHTS.closed_profile(tmp_path, rpc, identities)
    else:
        blob, sha = WEIGHTS.closed_profile(tmp_path, rpc, identities)
        assert hashlib.sha256(blob).hexdigest() == sha
        profile = SERVICE.parse(blob)
        assert (
            profile["closed_download"]["controller_json"].encode()
            == originals["controller-receipt.json"]
        )
        assert (
            profile["closed_download"]["result_json"].encode() == originals["download-result.json"]
        )
        # Warmed serving selects settings separately; it cannot rewrite the
        # original CPU closure or require the new serving source in old bytes.
        for workload in ("public-six", "known-four"):
            setting = SERVICE.startup_configuration("publisher-text48-v1", workload)
            assert setting["max_browser_envelopes"] in (64, 96)
            assert SERVICE.validate_weight_profile(blob, sha, identities) == profile
            assert hashlib.sha256(blob).hexdigest() == sha
            assert "startup" not in SERVICE.parse(blob)


def test_sdk_download_is_cpu_only_named_mounts_without_input_data_or_public_endpoint():
    tree = ast.parse(PATH.read_bytes())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "download"
    )
    options = {item.arg: item.value for item in function.decorator_list[0].keywords}
    assert "gpu" not in options and ast.literal_eval(options["memory"]) == 8192
    assert (
        ast.literal_eval(options["retries"]) == 0
        and ast.literal_eval(options["include_source"]) is False
    )
    assert [ast.literal_eval(key) for key in options["volumes"].keys] == ["/weights", "/export"]
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"materialize", "download_child", "asgi_app", "web_server"}
        for node in ast.walk(tree)
    )


def test_owned_download_creates_fresh_directory_with_real_file_only_core_api(tmp_path, monkeypatch):
    """Reach the child boundary using the real core input-file validator."""
    output = tmp_path / "public-weights-fresh-test"
    with pytest.raises(FileNotFoundError):
        CORE.ordinary(output)
    core = SimpleNamespace(
        OUTPUT_ROOT=tmp_path,
        ordinary=CORE.ordinary,
        write=lambda path, value, **_kwargs: SERVICE.save(path, value),
        authenticate_source=lambda _sha: {},
    )

    def tree_inventory(path):
        return {entry.name: SERVICE.pin(entry) for entry in path.iterdir() if entry.is_file()}

    staging = SimpleNamespace(
        capture_mount=lambda alias: {"alias": alias},
        child_environment=lambda _mode: {"CUDA_VISIBLE_DEVICES": ""},
        tree_inventory=tree_inventory,
        publish_output=lambda path, _binding, confirmed: (
            SERVICE.pin(path / "export-index.json") if confirmed else None
        ),
    )
    launches = []

    def launch(*args, **kwargs):
        launches.append((args, kwargs))
        assert output.is_dir() and (output / "source-inventory.json").is_file()
        raise OSError("FAKE_CHILD_LAUNCH_FAILURE")

    monkeypatch.setattr(WEIGHTS, "setup", lambda _blob: (SERVICE, staging, core, {}))
    monkeypatch.setattr(WEIGHTS, "authenticate_manifest", lambda _blob: None)
    monkeypatch.setattr(WEIGHTS, "subprocess", SimpleNamespace(Popen=launch, PIPE=-1))
    for name in ("weights", "results"):
        monkeypatch.setattr(WEIGHTS, name, SimpleNamespace(commit=lambda: None), raising=False)
    result = WEIGHTS.owned("fresh-test", b"{}")
    assert len(launches) == 1 and output.is_dir()
    assert result["success"] is False and result["owned_exit_confirmed"] is True
    assert result["error"]["message"] == "FAKE_CHILD_LAUNCH_FAILURE"
    assert result["export_index"] == SERVICE.pin(output / "export-index.json")
    controller = json.loads((output / "controller-receipt.json").read_bytes())
    assert controller["child_success"] is False and controller["model_loads"] == 0
