import pytest

from s1.runtime import resolve_device, resolve_dtype

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.inference


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_priority(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert resolve_device() == expected
    assert resolve_device("auto") == expected
    assert resolve_device("cpu") == "cpu"
    assert resolve_device(torch.device("cuda:1")) == "cuda:1"


@pytest.mark.parametrize("modern_macos", [True, False])
def test_auto_dtype_follows_device_and_mps_os_support(monkeypatch, modern_macos):
    calls = []

    def is_macos_or_newer(major, minor):
        calls.append((major, minor))
        return modern_macos

    monkeypatch.setattr(torch.backends.mps, "is_macos_or_newer", is_macos_or_newer)
    assert resolve_dtype("auto", "cpu") == torch.float32
    assert resolve_dtype(None, "cuda:1") == torch.bfloat16
    assert calls == []
    assert resolve_dtype("auto", "mps") == (torch.bfloat16 if modern_macos else torch.float32)
    assert calls == [(14, 0)]


@pytest.mark.parametrize("name", ["float32", "float16", "bfloat16"])
def test_explicit_dtype_is_preserved(name):
    dtype = getattr(torch, name)
    for device in ("cpu", "cuda", "mps"):
        assert resolve_dtype(name, device) == dtype
        assert resolve_dtype(dtype, device) == dtype


@pytest.mark.parametrize("dtype", ["float64", "invalid", torch.float64, 16, True])
def test_unsupported_dtype_rejected(dtype):
    with pytest.raises(ValueError, match="dtype must be"):
        resolve_dtype(dtype, "cpu")
