"""Read GPU identity/totals without creating a CUDA context or loading a model."""

import ctypes as c
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import uuid


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
resource.setrlimit(resource.RLIMIT_CPU, (10, 10))


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def total():
    raw = next(x for x in Path("/proc/meminfo").read_text().splitlines() if x.startswith("MemTotal:"))
    return {"raw_line": raw, "bytes": int(raw.split()[1]) * 1024}


def ref(path):
    raw = path.read_bytes()
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}


def main():
    started, boot = utc(), Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    host = total()
    n = c.CDLL("libnvidia-ml.so.1")
    assert n.nvmlInit_v2() == 0
    nvml = {}
    try:
        h = c.c_void_p()
        fn = n.nvmlDeviceGetHandleByIndex_v2
        fn.argtypes = [c.c_uint, c.POINTER(c.c_void_p)]
        assert fn(0, c.byref(h)) == 0

        class M(c.Structure):
            _fields_ = [("total", c.c_ulonglong), ("free", c.c_ulonglong), ("used", c.c_ulonglong)]

        m = M()
        fn = n.nvmlDeviceGetMemoryInfo
        fn.argtypes = [c.c_void_p, c.POINTER(M)]
        rc = fn(h, c.byref(m))
        nvml["memory_info_v1"] = {"returncode": rc, "valid_measurement": rc == 0}
        if rc == 0:
            nvml["memory_info_v1"]["total_bytes"] = m.total
        else:
            n.nvmlErrorString.restype = c.c_char_p
            nvml["memory_info_v1"]["error"] = n.nvmlErrorString(rc).decode()

        class M2(c.Structure):
            _fields_ = [("version", c.c_uint), ("total", c.c_ulonglong), ("reserved", c.c_ulonglong), ("free", c.c_ulonglong), ("used", c.c_ulonglong)]

        m2 = M2()
        m2.version = c.sizeof(M2) | (2 << 24)
        fn = n.nvmlDeviceGetMemoryInfo_v2
        fn.argtypes = [c.c_void_p, c.POINTER(M2)]
        rc = fn(h, c.byref(m2))
        nvml["memory_info_v2"] = {"returncode": rc, "valid_measurement": rc == 0}
        if rc == 0:
            nvml["memory_info_v2"]["total_bytes"] = m2.total
        else:
            nvml["memory_info_v2"]["error"] = n.nvmlErrorString(rc).decode()
    finally:
        nvml["shutdown_returncode"] = n.nvmlShutdown()

    driver_path = Path("/usr/lib/aarch64-linux-gnu/libcuda.so.1")
    lib = c.CDLL(str(driver_path))
    codes = {}

    def call(name, types, args):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = types, c.c_int
        rc = fn(*args)
        codes[name] = rc
        assert rc == 0, (name, rc)

    call("cuInit", [c.c_uint], [0])
    ctx = c.c_void_p()
    call("cuCtxGetCurrent", [c.POINTER(c.c_void_p)], [c.byref(ctx)])
    context_before = ctx.value
    count, device = c.c_int(), c.c_int()
    call("cuDeviceGetCount", [c.POINTER(c.c_int)], [c.byref(count)])
    call("cuDeviceGet", [c.POINTER(c.c_int), c.c_int], [c.byref(device), 0])
    memory = c.c_size_t()
    call("cuDeviceTotalMem_v2", [c.POINTER(c.c_size_t), c.c_int], [c.byref(memory), device])
    name, integrated = c.create_string_buffer(128), c.c_int()
    call("cuDeviceGetName", [c.c_char_p, c.c_int, c.c_int], [name, len(name), device])
    call("cuDeviceGetAttribute", [c.POINTER(c.c_int), c.c_int, c.c_int], [c.byref(integrated), 18, device])

    class U(c.Structure):
        _fields_ = [("bytes", c.c_ubyte * 16)]

    identity = U()
    call("cuDeviceGetUuid_v2", [c.POINTER(U), c.c_int], [c.byref(identity), device])
    call("cuCtxGetCurrent", [c.POINTER(c.c_void_p)], [c.byref(ctx)])
    assert context_before is None and ctx.value is None
    assert total() == host and Path("/proc/sys/kernel/random/boot_id").read_text().strip() == boot
    pins_path = ROOT / "artifacts/jev-runtime/public-native-fit-memory-peak-probe-20261018-01/native-toolchain-runtime-pins-02.json"
    pins = json.loads(pins_path.read_bytes())["cuda_link_origins"]
    origins = []
    for spec in pins:
        path = Path(spec["path"]).resolve(strict=True)
        st = path.stat()
        actual = {"resolved": str(path), "size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "device": st.st_dev, "inode": st.st_ino}
        expected = {k: spec[k] for k in actual}
        origins.append({"path": spec["path"], "actual": actual, "expected": expected, "matches": actual == expected})
    assert all(x["matches"] for x in origins)
    receipt = {
        "schema_id": "gx10_reboot_read_only_driver_identity_v1", "status": "PASS",
        "started_at_utc": started, "finished_at_utc": utc(), "boot_id": boot,
        "host_memtotal": host, "driver": {"device_count": count.value, "device": device.value, "total_bytes": memory.value, "name": name.value.decode(), "integrated": integrated.value, "uuid": str(uuid.UUID(bytes=bytes(identity.bytes))), "returncodes": codes},
        "context_before_device_queries": context_before, "context_after_device_queries": ctx.value,
        "historical_total_bytes": 130595860480, "observed_total_delta_bytes": memory.value - 130595860480,
        "raw_host_driver_totals_match": host["bytes"] == memory.value,
        "nvml": nvml, "driver_origins": origins, "driver_pin_descriptor": ref(pins_path),
        "entry": ref(Path(__file__).resolve()),
        "limits": {"address_space_bytes": None, "cpu_seconds": 10},
        "preparation_attempts": [
            {"address_space_bytes": 128 * 1024**2, "result": "libcuda mapping refused before cuInit"},
            {"address_space_bytes": 256 * 1024**2, "result": "cuInit returned2 before any CUDA context"},
        ],
        "model_loaded": False, "torch_imported": False, "cuda_context_created": False,
        "Comfy_mutations": 0, "scope": "Read-only metadata, not allocator capacity, CUDA execution, warmup or training qualification; no DSO payload hash claimed.",
    }
    assert receipt["raw_host_driver_totals_match"]
    path = HERE / "metadata-01.json"
    with path.open("x") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(ref(path)))


if __name__ == "__main__":
    main()
