"""Unit property admission tests; no unit, processor, CUDA or model is started."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "cpu3g_unit_test", Path(__file__).with_name("unit_runner.py")
)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def properties():
    return {
        "LoadState": "loaded",
        "ControlGroup": M.CGROUP,
        "Type": "exec",
        "MainPID": "1234",
        "MemoryAccounting": "yes",
        "MemoryMax": str(3 * M.GIB),
        "MemoryHigh": "infinity",
        "MemorySwapMax": "0",
        "OOMPolicy": "kill",
        "KillMode": "control-group",
        "RuntimeMaxUSec": "11min",
        "TimeoutStopUSec": "10s",
        "InvocationID": "a" * 32,
    }


def test_exact_own_unit():
    M.validate_properties(properties(), 1234)


@pytest.mark.parametrize(
    "key,value",
    [
        ("ControlGroup", "/user.slice/user-1001.slice/user@1001.service/app.slice"),
        ("MainPID", "1235"),
        ("MemoryMax", str(10 * M.GIB)),
        ("MemorySwapMax", "infinity"),
        ("MemoryAccounting", "no"),
        ("KillMode", "process"),
        ("OOMPolicy", "continue"),
        ("RuntimeMaxUSec", "infinity"),
        ("InvocationID", "z" * 32),
    ],
)
def test_refuse_scope_or_limit_drift(key, value):
    observed = properties()
    observed[key] = value
    with pytest.raises(ValueError):
        M.validate_properties(observed, 1234)
