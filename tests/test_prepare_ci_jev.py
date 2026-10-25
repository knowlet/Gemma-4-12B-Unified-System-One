"""Provisioning verifies pinned bytes without loading a model or accepting evidence."""

import hashlib
import io

import prepare_ci_jev as setup
import pytest


def test_fetches_and_reuses_verified_assets(tmp_path, monkeypatch):
    blob = b"synthetic tokenizer bytes"
    monkeypatch.setattr(setup, "ASSET_SHA", {"tokenizer.json": hashlib.sha256(blob).hexdigest()})
    calls = []

    def download(url, *, timeout):
        calls.append((url, timeout))
        return io.BytesIO(blob)

    monkeypatch.setattr(setup.urllib.request, "urlopen", download)
    setup.prepare(tmp_path)
    setup.prepare(tmp_path)
    assert (tmp_path / "tokenizer.json").read_bytes() == blob
    assert calls == [(f"{setup.BASE}/tokenizer.json", 60)]


@pytest.mark.parametrize("cached", [False, True])
def test_refuses_corrupt_assets_without_replacing_them(tmp_path, monkeypatch, cached):
    monkeypatch.setattr(setup, "ASSET_SHA", {"tokenizer.json": "0" * 64})
    path = tmp_path / "tokenizer.json"
    if cached:
        path.write_bytes(b"corrupt")
    monkeypatch.setattr(
        setup.urllib.request, "urlopen", lambda *_args, **_kw: io.BytesIO(b"corrupt")
    )
    with pytest.raises(ValueError, match="Pinned tokenizer asset differs"):
        setup.prepare(tmp_path)
    assert path.read_bytes() == b"corrupt" if cached else not path.exists()
