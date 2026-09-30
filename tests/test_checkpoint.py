"""Calibration loading tests; no model weights or network access required."""

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from s1.checkpoint import load_temperature


@pytest.fixture
def hub(monkeypatch):
    """Keep the remote contract testable without the inference extra."""
    module = ModuleType("huggingface_hub")
    errors = ModuleType("huggingface_hub.errors")

    class EntryNotFoundError(Exception):
        pass

    class RemoteEntryNotFoundError(EntryNotFoundError):
        pass

    class LocalEntryNotFoundError(FileNotFoundError, EntryNotFoundError):
        pass

    errors.RemoteEntryNotFoundError = RemoteEntryNotFoundError
    errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    download = Mock(side_effect=AssertionError("unexpected Hub download"))
    module.hf_hub_download = download
    module.errors = errors
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors)
    return SimpleNamespace(download=download, errors=errors)


@pytest.mark.parametrize("temperature", [0.25, 2, 1e-100, 1e100])
@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
def test_load_temperature(tmp_path, hub, temperature, remote):
    path = tmp_path / "s1_config.json"
    path.write_text(json.dumps({"temperature": temperature, "prompt_version": 1}))
    hub.download.side_effect = None
    hub.download.return_value = str(path)

    result = load_temperature("owner/checkpoint" if remote else tmp_path)

    assert result == temperature
    assert isinstance(result, float)
    if remote:
        hub.download.assert_called_once_with("owner/checkpoint", "s1_config.json", revision=None)
    else:
        hub.download.assert_not_called()


def test_local_loading_does_not_require_hub(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", None)
    assert load_temperature(str(tmp_path), revision="unused-for-local") == 1.0
    (tmp_path / "s1_config.json").write_text('{"temperature": 2.5}')
    assert load_temperature(str(tmp_path), revision="unused-for-local") == 2.5


@pytest.mark.parametrize("revision", ["a" * 40, "release-tag", "moving-branch", None])
def test_remote_uses_exact_supplied_revision(tmp_path, hub, revision):
    path = tmp_path / "s1_config.json"
    path.write_text('{"temperature": 2.5}')
    hub.download.side_effect = None
    hub.download.return_value = str(path)

    assert load_temperature("owner/checkpoint", revision=revision) == 2.5

    hub.download.assert_called_once_with("owner/checkpoint", "s1_config.json", revision=revision)


def test_remote_missing_optional_file_defaults(hub):
    hub.download.side_effect = hub.errors.RemoteEntryNotFoundError("file absent at commit")
    assert load_temperature("owner/checkpoint", revision="a" * 40) == 1.0
    hub.download.assert_called_once_with("owner/checkpoint", "s1_config.json", revision="a" * 40)


def test_remote_offline_cache_miss_is_not_missing_optional_file(hub):
    error = hub.errors.LocalEntryNotFoundError("offline; file not cached")
    hub.download.side_effect = error
    with pytest.raises(type(error)) as caught:
        load_temperature("owner/checkpoint", revision="a" * 40)
    assert caught.value is error


@pytest.mark.parametrize(
    "error_type",
    [PermissionError, ConnectionError, TimeoutError, FileNotFoundError, OSError, ValueError],
)
def test_remote_download_errors_propagate(hub, error_type):
    error = error_type("download unavailable")
    hub.download.side_effect = error
    with pytest.raises(error_type) as caught:
        load_temperature("owner/checkpoint", revision="a" * 40)
    assert caught.value is error
    assert hub.download.call_count == 1


def test_missing_downloaded_cache_file_is_an_error(tmp_path, hub):
    hub.download.side_effect = None
    hub.download.return_value = str(tmp_path / "s1_config.json")
    with pytest.raises(FileNotFoundError):
        load_temperature("owner/checkpoint")


@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
@pytest.mark.parametrize(
    "content",
    ["", "{", '{"temperature": 2,}', "not JSON"],
)
def test_malformed_json_is_an_error(tmp_path, hub, remote, content):
    path = tmp_path / "s1_config.json"
    path.write_text(content)
    hub.download.side_effect = None
    hub.download.return_value = str(path)
    with pytest.raises(json.JSONDecodeError):
        load_temperature("owner/checkpoint" if remote else tmp_path)


@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
@pytest.mark.parametrize("config", [{}, [], None, 2, "temperature", {"other": 2}])
def test_missing_temperature_or_invalid_config_is_an_error(tmp_path, hub, remote, config):
    path = tmp_path / "s1_config.json"
    path.write_text(json.dumps(config))
    hub.download.side_effect = None
    hub.download.return_value = str(path)
    with pytest.raises(ValueError, match="s1_config.json.*containing temperature"):
        load_temperature("owner/checkpoint" if remote else tmp_path)


@pytest.mark.parametrize("remote", [False, True], ids=["local", "remote"])
@pytest.mark.parametrize(
    "temperature",
    [0, -1, float("nan"), float("inf"), -float("inf"), True, False, "2", None, [], {}, 10**400],
    ids=[
        "zero",
        "negative",
        "nan",
        "inf",
        "-inf",
        "true",
        "false",
        "str",
        "null",
        "list",
        "dict",
        "overflow",
    ],
)
def test_invalid_temperature_is_an_error(tmp_path, hub, remote, temperature):
    path = tmp_path / "s1_config.json"
    path.write_text(json.dumps({"temperature": temperature}))
    hub.download.side_effect = None
    hub.download.return_value = str(path)
    with pytest.raises(ValueError, match="s1_config.json.*positive, finite number"):
        load_temperature("owner/checkpoint" if remote else tmp_path)


def test_local_config_directory_is_an_error(tmp_path, hub):
    (tmp_path / "s1_config.json").mkdir()
    with pytest.raises(IsADirectoryError):
        load_temperature(tmp_path)
    hub.download.assert_not_called()


def test_local_broken_config_symlink_is_an_error(tmp_path, hub):
    (tmp_path / "s1_config.json").symlink_to(tmp_path / "missing-target")
    with pytest.raises(FileNotFoundError):
        load_temperature(tmp_path)
    hub.download.assert_not_called()


@pytest.mark.parametrize("operation", ["stat", "lstat", "read_text"])
def test_local_access_errors_propagate(tmp_path, hub, monkeypatch, operation):
    (tmp_path / "s1_config.json").write_text('{"temperature": 2}')
    error = PermissionError("access denied")
    with monkeypatch.context() as patch:
        patch.setattr(Path, operation, Mock(side_effect=error))
        with pytest.raises(PermissionError) as caught:
            load_temperature(tmp_path)
    assert caught.value is error
    hub.download.assert_not_called()


@pytest.mark.parametrize(
    "error_name",
    [
        "RemoteEntryNotFoundError",
        "LocalEntryNotFoundError",
        "GatedRepoError",
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "HfHubHTTPError",
        "OfflineModeIsEnabled",
    ],
)
def test_real_hub_error_types(monkeypatch, error_name):
    """Also exercise the installed Hub hierarchy when inference is available."""
    module = pytest.importorskip("huggingface_hub")
    errors = pytest.importorskip("huggingface_hub.errors")
    import httpx

    error_type = getattr(errors, error_name)
    if issubclass(error_type, errors.HfHubHTTPError):
        response = httpx.Response(
            404, request=httpx.Request("HEAD", "https://hub.invalid/s1_config.json")
        )
        error = error_type("Hub failure", response=response)
    else:
        error = error_type("Hub failure")
    download = Mock(side_effect=error)
    monkeypatch.setattr(module, "hf_hub_download", download)

    if error_name == "RemoteEntryNotFoundError":
        assert load_temperature("owner/checkpoint", revision="a" * 40) == 1.0
    else:
        with pytest.raises(type(error)) as caught:
            load_temperature("owner/checkpoint", revision="a" * 40)
        assert caught.value is error
    download.assert_called_once_with("owner/checkpoint", "s1_config.json", revision="a" * 40)
