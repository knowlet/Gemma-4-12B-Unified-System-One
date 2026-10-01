"""Repeated cloud invocations isolate and clean staging without starting Modal."""

import importlib.metadata
import importlib.util
import io
import json
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from zipfile import ZipFile

import pytest


@pytest.fixture
def preparation(monkeypatch, tmp_path):
    class Image:
        def __getattr__(self, name):
            return lambda *args, **kwargs: self

    def decorate(**kwargs):
        return lambda function: function

    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            is_local=lambda: True,
            Image=SimpleNamespace(debian_slim=lambda **kwargs: Image()),
            App=lambda *args: SimpleNamespace(function=decorate, local_entrypoint=decorate),
        ),
    )
    path = Path(__file__).resolve().parents[1] / "apps/modal/prepare_comparison_data.py"
    spec = importlib.util.spec_from_file_location("comparison_data_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "source.txt").write_text("original")
    datasets = {"text-test.jsonl": {"dataset_sha256": "a" * 64}}
    historical = tmp_path / "historical.json"
    historical.write_text(json.dumps({"data": {"datasets": datasets}}))
    state = {"fail": False, "destinations": []}

    def prepare_data(destination):
        state["destinations"].append(destination)
        assert (destination / "raw/source.txt").read_text() == "original"
        if state["fail"]:
            raise ValueError("fixture preparation failed")
        (destination / "text-test.jsonl").write_text('{"id":"fixture"}\n')
        return {"datasets": datasets, "media_bundle_sha256": "b" * 64}

    def mapped_path(value):
        if str(value) == "/workspace/historical.json":
            return historical
        if str(value) == "/tmp/recovered-datasets":
            return tmp_path / "legacy-staging"
        return Path(value)

    monkeypatch.setattr(module, "Path", mapped_path)
    monkeypatch.setattr(
        module,
        "shutil",
        SimpleNamespace(copytree=lambda source, target: shutil.copytree(raw, target)),
    )
    monkeypatch.setattr(
        module, "TemporaryDirectory", lambda **kwargs: TemporaryDirectory(dir=tmp_path, **kwargs)
    )
    monkeypatch.setitem(
        sys.modules, "prepare_live_validation_data", SimpleNamespace(prepare=prepare_data)
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(sys, "path", list(sys.path))
    return module, state


def test_preparation_can_run_twice_in_same_container(preparation):
    module, state = preparation
    for _ in range(2):
        result = module.prepare()
        assert result["diagnostics"]["matches"] == {"text-test.jsonl": True}
        assert not state["destinations"][-1].exists()
        with ZipFile(io.BytesIO(result["archive"])) as archive:
            assert set(archive.namelist()) == {"text-test.jsonl", "recovery.json"}
            assert json.loads(archive.read("recovery.json")) == result["diagnostics"]
    assert state["destinations"][0] != state["destinations"][1]


def test_failed_preparation_cleans_staging_before_retry(preparation):
    module, state = preparation
    state["fail"] = True
    with pytest.raises(ValueError, match="fixture preparation failed"):
        module.prepare()
    assert not state["destinations"][-1].exists()
    state["fail"] = False
    assert module.prepare()["diagnostics"]["matches"] == {"text-test.jsonl": True}
    assert not state["destinations"][-1].exists()
    assert state["destinations"][0] != state["destinations"][1]
