"""UMA policy deltas with fake Driver/Torch only; no library load or GPU query.

Use the existing NVIDIA/Linux UMA research and H13 source audit. These tests
check identity, raw-counter boundaries and durable admission evidence rather
than running or copying the historical native-training regression suite.
"""

import copy
import ctypes
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).with_name("native_uma_peak_probe_01.py")
UUID = "00112233-4455-6677-8899-aabbccddeeff"
BOOT = "aabbccdd-0011-2233-4455-66778899aabb"


@pytest.fixture
def probe(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("A pure test attempted to load a native library")

    monkeypatch.setattr(ctypes, "CDLL", forbidden)
    spec = importlib.util.spec_from_file_location("isolated_uma_delta_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "read", forbidden)
    return module


@pytest.fixture
def case(probe):
    host = {
        "MemTotal": probe.UMA_TOTAL,
        "MemAvailable": 120_000_000_000,
        "MemFree": 90_000_000_000,
        "Cached": 20_000_000_000,
        "SReclaimable": 1_000_000_000,
        "SwapTotal": 16 * probe.GIB,
        "SwapFree": 15 * probe.GIB,
        "Hugepagesize": 2 * probe.MIB,
        "Hugetlb": 0,
        "HugePages_Total": 0,
        "HugePages_Free": 0,
        "boot_id": BOOT,
    }
    allocator = {
        "name": "NVIDIA GB10",
        "uuid": "GPU-" + UUID,
        "total_bytes": probe.UMA_TOTAL,
        "fraction": 0.3,
    }
    identity = {
        "context_device": 3,
        "integrated": 1,
        "name": "NVIDIA GB10",
        "uuid": UUID.replace("-", "").upper(),
        "total_bytes": probe.UMA_TOTAL,
        "boot_id": BOOT,
    }
    cuda = {
        "free": 39 * probe.GIB,
        "total": probe.UMA_TOTAL,
        "reserved": 28 * probe.GIB,
        "backend": "native",
        "fraction": 0.3,
    }
    return SimpleNamespace(
        host=host,
        allocator=allocator,
        identity=identity,
        cuda=cuda,
        config={"boot_id": BOOT, "device_uuid": "gpu-" + UUID.upper()},
    )


def decision(probe, case):
    return probe.uma_memory_decision(
        case.host, case.cuda, case.identity, case.allocator, case.config
    )


def test_meminfo_parser_keeps_hugetlb_counts_separate_from_kib(probe, case, monkeypatch):
    lines = [
        f"{key}: {value if key.startswith('HugePages_') else value // 1024}"
        + ("" if key.startswith("HugePages_") else " kB")
        for key, value in case.host.items()
        if key != "boot_id"
    ]
    monkeypatch.setattr(probe, "read", lambda *_: "\n".join(lines).encode())
    monkeypatch.setattr(probe, "boot_id", lambda: BOOT)
    observed = probe.host_memory()
    assert observed["HugePages_Total"] == observed["Hugetlb"] == 0
    assert observed["Hugepagesize"] == 2 * probe.MIB
    assert observed["MemTotal"] == probe.UMA_TOTAL and observed["boot_id"] == BOOT
    probe.uma_host_identity(observed, case.config)


@pytest.mark.parametrize(
    "field",
    [
        "MemTotal",
        "MemAvailable",
        "MemFree",
        "Cached",
        "SReclaimable",
        "SwapTotal",
        "SwapFree",
        "Hugepagesize",
        "Hugetlb",
        "HugePages_Total",
        "HugePages_Free",
    ],
)
@pytest.mark.parametrize("bad", [None, True, -1, "0"])
def test_host_identity_refuses_missing_or_untyped_evidence(probe, case, field, bad):
    if bad is None:
        case.host.pop(field)
    else:
        case.host[field] = bad
    with pytest.raises((ValueError, KeyError)):
        probe.uma_host_identity(case.host, case.config)


@pytest.mark.parametrize(
    "changes",
    [
        {"MemAvailable": 0},
        {"MemAvailable": 130595860481},
        {"MemFree": 130595860481},
        {"MemTotal": 130595860479},
        {"SwapFree": 17 * 1024**3},
        {"Hugepagesize": 0},
        {"HugePages_Total": 1},
        {"HugePages_Free": 1},
        {"Hugetlb": 4096},
        {"boot_id": "another-boot"},
    ],
)
def test_host_identity_refuses_inconsistent_or_unsupported_host(probe, case, changes):
    case.host.update(changes)
    with pytest.raises(ValueError):
        probe.uma_host_identity(case.host, case.config)


@pytest.mark.parametrize(
    "owner,field,bad",
    [
        ("identity", "integrated", True),
        ("identity", "integrated", 0),
        ("identity", "integrated", 1.0),
        ("identity", "name", "Other GPU"),
        ("identity", "total_bytes", 130595860479),
        ("identity", "total_bytes", True),
        ("identity", "boot_id", "another-boot"),
        ("identity", "uuid", "f" * 32),
        ("allocator", "uuid", "f" * 32),
        ("config", "device_uuid", "f" * 32),
        ("identity", "uuid", True),
        ("identity", "uuid", "not-a-uuid"),
        ("allocator", "name", "Other GPU"),
        ("allocator", "total_bytes", 1),
    ],
)
def test_driver_identity_requires_same_torch_selected_device(probe, case, owner, field, bad):
    probe.uma_validate_identity(case.identity, case.allocator, case.host, case.config)
    getattr(case, owner)[field] = bad
    with pytest.raises(ValueError):
        probe.uma_validate_identity(case.identity, case.allocator, case.host, case.config)


@pytest.mark.parametrize("free", [0, 39 * 1024**3, 41 * 1024**3])
def test_uma_admission_preserves_raw_free_and_old_gate_outcome(probe, case, free):
    case.cuda["free"] = free
    before = copy.deepcopy(vars(case))
    result = decision(probe, case)
    assert result["UMA_predicate_pass"] and result["native_allocator_predicate_pass"]
    assert result["original_cuda_free_would_refuse"] is (free < 41 * probe.GIB)
    assert result["original_native_device_guard_would_refuse"] is (free < 41 * probe.GIB)
    assert result["UMA_estimate_bytes"] == case.host["MemAvailable"]
    assert result["swap_credit_bytes"] == 0
    assert vars(case) == before


@pytest.mark.parametrize(
    "field,bad",
    [
        ("free", True),
        ("free", 1.0),
        ("free", -1),
        ("free", 130595860481),
        ("total", True),
        ("total", 130595860479),
        ("total", -1),
        ("reserved", True),
        ("reserved", -1),
        ("reserved", "0"),
        ("free", None),
        ("total", None),
        ("reserved", None),
    ],
)
def test_uma_refuses_bad_raw_counters_without_reclassifying_them(probe, case, field, bad):
    if bad is None:
        case.cuda.pop(field)
    else:
        case.cuda[field] = bad
    before = copy.deepcopy(case.cuda)
    with pytest.raises((ValueError, KeyError)):
        decision(probe, case)
    assert case.cuda == before


@pytest.mark.parametrize(
    "changes",
    [
        {"backend": "cudaMallocAsync"},
        {"fraction": 0.4},
        {"reserved": 42412802048},
    ],
)
def test_uma_does_not_bypass_native_allocator_guard(probe, case, changes):
    case.cuda.update(free=50 * probe.GIB, **changes)
    result = decision(probe, case)
    assert result["UMA_predicate_pass"] and not result["original_cuda_free_would_refuse"]
    assert not result["native_allocator_predicate_pass"]
    assert result["original_native_device_guard_would_refuse"]


class FakeSymbol:
    def __init__(self, name, calls, failure=None, device=3):
        self.name, self.calls, self.failure, self.device = name, calls, failure, device

    def __call__(self, *args):
        self.calls.append((self.name, args))
        if self.failure is not None:
            return self.failure
        if self.name == "cuCtxGetDevice":
            args[0]._obj.value = self.device
        elif self.name == "cuDeviceGetAttribute":
            args[0]._obj.value = 1
        elif self.name == "cuDeviceGetUuid":
            args[0][:] = bytes.fromhex(UUID.replace("-", ""))
        elif self.name == "cuDeviceGetName":
            args[0].value = b"NVIDIA GB10"
        elif self.name == "cuDeviceTotalMem_v2":
            args[0]._obj.value = 130595860480
        return 0


def fake_symbols(probe, *, failed_name=None, failure=None, device=3):
    calls = []
    symbols = {
        name: FakeSymbol(name, calls, failure if name == failed_name else None, device)
        for name in probe.UMA_SYMBOLS
    }
    return symbols, calls


def test_driver_read_uses_current_context_attr18_and_exact_ctypes_signatures(probe):
    symbols, calls = fake_symbols(probe)
    actual = probe.uma_driver_read(symbols, ctypes)
    assert actual == {
        "context_device": 3,
        "integrated": 1,
        "uuid": UUID.replace("-", ""),
        "name": "NVIDIA GB10",
        "total_bytes": probe.UMA_TOTAL,
    }
    assert [name for name, _ in calls] == list(probe.UMA_SYMBOLS)
    assert calls[1][1][1:] == (18, 3)
    assert all(fn.restype is ctypes.c_int for fn in symbols.values())
    assert symbols["cuCtxGetDevice"].argtypes == [ctypes.POINTER(ctypes.c_int)]
    assert symbols["cuDeviceTotalMem_v2"].argtypes == [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_int,
    ]


@pytest.mark.parametrize(
    "name",
    [
        "cuCtxGetDevice",
        "cuDeviceGetAttribute",
        "cuDeviceGetUuid",
        "cuDeviceGetName",
        "cuDeviceTotalMem_v2",
    ],
)
@pytest.mark.parametrize("failure", [False, 1, "0"])
def test_driver_read_rejects_non_native_success_and_stops_at_failed_call(probe, name, failure):
    symbols, calls = fake_symbols(probe, failed_name=name, failure=failure)
    with pytest.raises(ValueError, match="Driver identity query failed"):
        probe.uma_driver_read(symbols, ctypes)
    assert calls[-1][0] == name
    assert len(calls) == list(probe.UMA_SYMBOLS).index(name) + 1


def test_driver_read_refuses_missing_current_context(probe):
    symbols, calls = fake_symbols(probe, device=-1)
    with pytest.raises(ValueError, match="Current CUDA context device"):
        probe.uma_driver_read(symbols, ctypes)
    assert len(calls) == 1


@pytest.fixture
def mappings(probe):
    metadata = {"resolved": probe.UMA_LIBCUDA, "device": os.makedev(8, 2), "inode": 77}
    raw = f"1000-2000 r-xp 00000000 08:02 77 {probe.UMA_LIBCUDA}\n"
    return SimpleNamespace(
        metadata=metadata,
        raw=raw,
        addresses={name: 0x1100 + i for i, name in enumerate(probe.UMA_SYMBOLS)},
    )


def test_all_driver_symbols_must_belong_to_same_exact_executable_inode(probe, mappings):
    result = probe.uma_symbol_mappings(mappings.raw, mappings.addresses, mappings.metadata)
    assert result == mappings.addresses
    assert result is not mappings.addresses


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "zero",
        "bool",
        "outside",
        "nonexec",
        "path",
        "device",
        "inode",
        "deleted",
    ],
)
def test_symbol_origin_rejects_incomplete_or_foreign_mappings(probe, mappings, mutation):
    first = probe.UMA_SYMBOLS[0]
    if mutation == "missing":
        mappings.addresses.pop(first)
    elif mutation == "extra":
        mappings.addresses["unexpected"] = 0x1100
    elif mutation in {"zero", "bool", "outside"}:
        mappings.addresses[first] = {"zero": 0, "bool": True, "outside": 0x2000}[mutation]
    else:
        old, new = {
            "nonexec": ("r-xp", "r--p"),
            "path": (probe.UMA_LIBCUDA, "/other/libcuda.so.1"),
            "device": ("08:02", "08:03"),
            "inode": (" 77 ", " 78 "),
            "deleted": (probe.UMA_LIBCUDA, probe.UMA_LIBCUDA + " (deleted)"),
        }[mutation]
        mappings.raw = mappings.raw.replace(old, new)
    with pytest.raises(ValueError):
        probe.uma_symbol_mappings(mappings.raw, mappings.addresses, mappings.metadata)


@pytest.fixture
def driver_wrapper(probe, mappings, monkeypatch):
    trace, control = [], {"fault": None, "metadata": 0, "maps": 0}
    symbols, calls = fake_symbols(probe)
    metadata = {**mappings.metadata, "size_bytes": 96_460_560, "mtime_ns": 1234}
    runtime = {"synthetic_runtime": True}
    expected_mode = os.RTLD_NOLOAD | os.RTLD_NOW | os.RTLD_LOCAL

    class TracedSymbol:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            trace.append("query:" + self.name)
            return symbols[self.name](*args)

    def fake_cdll(path, *, mode):
        assert path == probe.UMA_LIBCUDA and mode == expected_mode
        trace.append("load")
        return SimpleNamespace(**{name: TracedSymbol(name) for name in probe.UMA_SYMBOLS})

    def fake_cast(symbol, target_type):
        assert target_type is ctypes.c_void_p
        trace.append("cast:" + symbol.name)
        return SimpleNamespace(value=mappings.addresses[symbol.name])

    def observed_metadata(value):
        assert value is runtime
        trace.append("metadata")
        control["metadata"] += 1
        result = dict(metadata)
        if control["fault"] == "metadata" and control["metadata"] == 2:
            result["inode"] += 1
        return result

    def observed_maps(path, maximum):
        assert str(path) == f"/proc/{os.getpid()}/maps" and maximum == 8 * probe.MIB
        trace.append("maps")
        control["maps"] += 1
        raw = mappings.raw
        if control["fault"] == "initial-map" or (
            control["fault"] == "final-map" and control["maps"] == 2
        ):
            raw = raw.replace(" 77 ", " 78 ")
        return raw.encode()

    def observed_boot():
        trace.append("boot")
        return "changed-boot" if control["fault"] == "boot" else BOOT

    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(ctypes, "cast", fake_cast)
    monkeypatch.setattr(probe, "uma_driver_metadata", observed_metadata)
    monkeypatch.setattr(probe, "read", observed_maps)
    monkeypatch.setattr(probe, "boot_id", observed_boot)
    return SimpleNamespace(
        runtime=runtime,
        metadata=metadata,
        trace=trace,
        control=control,
        symbols=symbols,
        calls=calls,
        addresses=mappings.addresses,
    )


def test_identity_wrapper_binds_noload_context_and_rechecks_metadata_maps_boot(
    probe, case, driver_wrapper
):
    fake = driver_wrapper
    before = copy.deepcopy(vars(case))
    actual = probe.uma_driver_identity(fake.runtime, case.allocator, case.host, case.config)
    assert fake.trace == (
        ["metadata", "load"]
        + ["cast:" + name for name in probe.UMA_SYMBOLS]
        + ["maps"]
        + ["query:" + name for name in probe.UMA_SYMBOLS]
        + ["metadata", "maps", "boot"]
    )
    assert actual["context_device"] == 3 and actual["integrated"] == 1
    assert actual["uuid"] == UUID.replace("-", "") and actual["boot_id"] == BOOT
    assert actual["profile_kind"] == probe.UMA_KIND
    assert actual["library_metadata"] == fake.metadata
    assert actual["symbol_addresses"] == fake.addresses
    assert actual["load_mode"] == "exact-canonical-RTLD_NOLOAD"
    assert fake.calls[1][1][1:] == (18, 3)
    assert vars(case) == before


@pytest.mark.parametrize(
    "fault,match,metadata_reads,map_reads,query_count",
    [
        ("unsupported", "already-loaded libcuda handle support", 1, 0, 0),
        ("initial-map", "Mapped libcuda path/dev/inode drift", 1, 1, 0),
        ("metadata", "Stable libcuda metadata after query", 2, 1, 5),
        ("query", "Driver identity query failed: cuDeviceGetAttribute", 1, 1, 2),
        ("startup", "unchanged startup floor", 0, 0, 0),
        ("final-map", "Mapped libcuda path/dev/inode drift", 2, 2, 5),
        ("boot", "Same boot after Driver query", 2, 2, 5),
    ],
)
def test_identity_wrapper_stops_at_failed_boundary_without_retry(
    probe, case, driver_wrapper, monkeypatch, fault, match, metadata_reads, map_reads, query_count
):
    fake = driver_wrapper
    fake.control["fault"] = fault
    if fault == "unsupported":
        monkeypatch.delattr(probe.os, "RTLD_NOLOAD")
    elif fault == "query":
        fake.symbols["cuDeviceGetAttribute"].failure = 1
    elif fault == "startup":
        case.host["MemAvailable"] = (
            probe.research_floor(case.host, startup=True)["required_bytes"] - 1
        )
    with pytest.raises(ValueError, match=match):
        probe.uma_driver_identity(fake.runtime, case.allocator, case.host, case.config)
    assert fake.control["metadata"] == metadata_reads
    assert fake.control["maps"] == map_reads and len(fake.calls) == query_count
    assert fake.trace.count("load") == (0 if fault in {"unsupported", "startup"} else 1)
    assert fake.trace.count("boot") == (1 if fault == "boot" else 0)


@pytest.mark.parametrize(
    "fault",
    [None, "missing-alias", "canonical", "mode", "size_bytes", "mtime_ns", "device", "inode"],
)
def test_driver_metadata_binds_both_aliases_to_same_canonical_inode_without_io(
    probe, monkeypatch, fault
):
    expected = {
        "resolved": probe.UMA_LIBCUDA,
        "size_bytes": 96_460_560,
        "mtime_ns": 1234,
        "device": os.makedev(8, 2),
        "inode": 77,
    }
    aliases = ["/lib/aarch64-linux-gnu/libcuda.so.1", "/usr/lib/aarch64-linux-gnu/libcuda.so.1"]
    origins = [{"path": path, **expected} for path in aliases]
    observed = []

    class FakePath:
        def __init__(self, path, *, alias=None):
            self.path, self.alias = path, alias

        def resolve(self, *, strict):
            assert strict is True and self.path in aliases
            observed.append(("resolve", self.path))
            return FakePath(probe.UMA_LIBCUDA, alias=self.path)

        def stat(self):
            observed.append(("stat", self.alias))
            values = {
                "st_mode": probe.stat.S_IFREG | 0o644,
                "st_size": expected["size_bytes"],
                "st_mtime_ns": expected["mtime_ns"],
                "st_dev": expected["device"],
                "st_ino": expected["inode"],
            }
            fields = {
                "size_bytes": "st_size",
                "mtime_ns": "st_mtime_ns",
                "device": "st_dev",
                "inode": "st_ino",
            }
            if self.alias == aliases[1] and fault in fields:
                values[fields[fault]] += 1
            if self.alias == aliases[1] and fault == "mode":
                values["st_mode"] = probe.stat.S_IFDIR | 0o755
            return SimpleNamespace(**values)

        def __str__(self):
            return self.path

    if fault == "missing-alias":
        origins.pop()
    elif fault == "canonical":
        origins[1]["resolved"] = "/different/libcuda.so.1"
    monkeypatch.setattr(probe, "Path", FakePath)
    if fault is not None:
        with pytest.raises(ValueError):
            probe.uma_driver_metadata({"cuda_link_origins": origins})
        if fault == "missing-alias":
            assert not observed
    else:
        assert probe.uma_driver_metadata({"cuda_link_origins": origins}) == expected
        assert observed == [
            (operation, alias) for alias in aliases for operation in ("resolve", "stat")
        ]


@pytest.mark.parametrize(
    "free,total,valid",
    [
        (0, 130595860480, True),
        (1, 130595860480, True),
        (False, 130595860480, False),
        (1.0, 130595860480, False),
        (-1, 130595860480, False),
        (130595860481, 130595860480, False),
        (0, True, False),
        (0, 130595860480.0, False),
        (0, 130595860479, False),
    ],
)
def test_cuda_query_keeps_zero_and_rejects_coercible_or_out_of_range_values(
    probe, free, total, valid
):
    calls = []

    def counter(*args):
        calls.append("counter")
        return 0

    cuda = SimpleNamespace(
        mem_get_info=lambda _: (free, total),
        memory_allocated=counter,
        memory_reserved=counter,
        max_memory_allocated=counter,
        max_memory_reserved=counter,
        memory_stats=lambda _: {},
        get_per_process_memory_fraction=lambda _: 0.3,
        memory=SimpleNamespace(get_allocator_backend=lambda: "native"),
    )
    instance = probe.Research.__new__(probe.Research)
    instance.torch = SimpleNamespace(cuda=cuda)
    instance.allocator = {"total_bytes": probe.UMA_TOTAL}
    if valid:
        result = instance.cuda()
        assert result["free"] == free and type(result["free"]) is int
        assert result["total"] == probe.UMA_TOTAL and len(calls) == 4
    else:
        with pytest.raises(ValueError):
            instance.cuda()
        assert not calls


@pytest.fixture
def gate(probe, case, tmp_path, monkeypatch):
    instance = probe.Research.__new__(probe.Research)
    instance.output, instance.config = tmp_path, case.config
    instance.binding, instance.allocator = {"ControlGroup": "/fake-owned"}, case.allocator
    instance.uma_identity, instance.loaded = case.identity, True
    cg = {
        "current": probe.GIB,
        "peak": probe.GIB,
        "swap": 0,
        "events": {},
        "max": probe.CAP,
        "high": None,
        "swap_max": 0,
        "oom_group": 1,
    }
    monkeypatch.setattr(probe, "host_memory", lambda: copy.deepcopy(case.host))
    monkeypatch.setattr(probe, "cgroup_snapshot", lambda _: copy.deepcopy(cg))
    monkeypatch.setattr(instance, "cuda", lambda: copy.deepcopy(case.cuda))
    return SimpleNamespace(instance=instance, cg=cg, path=tmp_path / "phases.jsonl")


def only_gate_row(gate):
    rows = [json.loads(line) for line in gate.path.read_text().splitlines()]
    assert len(rows) == 1
    return rows[0]


def test_gate_admits_low_cuda_free_and_durably_keeps_original_veto(probe, case, gate):
    assert gate.instance.check("native-backward", "research") is True
    row = only_gate_row(gate)
    assert row["status"] == "ADMITTED" and row["admitted"]
    assert row["cuda"] == case.cuda and row["observation"] == case.host
    assert row["original_cuda_free_would_refuse"]
    assert row["unchanged_research_host_floor_pass"] and row["unchanged_cgroup_pressure_pass"]
    assert row["resident_credit_bytes"] == row["swap_credit_bytes"] == 0


@pytest.mark.parametrize(
    "boundary,offset,admitted,uma_pass",
    [
        ("resident-host", -1, False, True),
        ("resident-host", 0, True, True),
        ("uma", -1, False, False),
        ("uma", 0, False, True),
        ("reserved", -1, True, True),
        ("reserved", 0, False, True),
    ],
)
def test_full_gate_keeps_higher_host_floor_and_exact_reserved_stop(
    probe, case, gate, boundary, offset, admitted, uma_pass
):
    resident_floor = probe.research_floor(case.host, startup=False)["required_bytes"]
    assert resident_floor == 77_658_779_648 > probe.CAP + probe.GIB
    if boundary == "resident-host":
        case.host["MemAvailable"] = resident_floor + offset
    elif boundary == "uma":
        case.host["MemAvailable"] = probe.CAP + probe.GIB + offset
    else:
        case.cuda["reserved"] = probe.STOP + offset
    if admitted:
        assert gate.instance.check("native-backward", "research") is True
    else:
        with pytest.raises(ValueError, match="admission refused"):
            gate.instance.check("native-backward", "research")
    row = only_gate_row(gate)
    assert row["admitted"] is admitted
    assert row["status"] == ("ADMITTED" if admitted else "REFUSED")
    assert row["UMA_predicate_pass"] is uma_pass
    assert row["required_bytes"] == resident_floor
    assert row["unchanged_research_host_floor_pass"] is (
        case.host["MemAvailable"] >= resident_floor
    )
    assert row["native_allocator_predicate_pass"] is (case.cuda["reserved"] < probe.STOP)
    assert row["cuda"] == case.cuda and row["observation"] == case.host
    assert row["original_cuda_free_would_refuse"] is True


@pytest.mark.parametrize("reason", ["host", "reserved", "swap", "max", "oom"])
def test_gate_refusal_is_durable_with_raw_evidence_before_raise(probe, case, gate, reason):
    if reason == "host":
        case.host["MemAvailable"] = (
            probe.research_floor(case.host, startup=False)["required_bytes"] - 1
        )
    elif reason == "reserved":
        case.cuda["reserved"] = probe.STOP
    else:
        gate.cg[{"swap": "swap", "max": "max", "oom": "events"}[reason]] = {
            "swap": 1,
            "max": probe.CAP - 1,
            "oom": {"oom": 1},
        }[reason]
    with pytest.raises(ValueError, match="admission refused"):
        gate.instance.check("native-Adam-step", "research")
    row = only_gate_row(gate)
    assert row["status"] == "REFUSED" and not row["admitted"]
    assert row["cuda"] == case.cuda and row["observation"] == case.host
    assert row["cgroup"] == gate.cg and row["original_cuda_free_would_refuse"]


@pytest.mark.parametrize("stage", ["host", "cgroup", "cuda"])
def test_query_error_is_unknown_not_false_pass_and_preserves_original_error(
    probe, case, gate, monkeypatch, stage
):
    original = RuntimeError("synthetic-" + stage + "-query")

    def query_failed(*args):
        raise original

    if stage == "cuda":
        monkeypatch.setattr(gate.instance, "cuda", query_failed)
    else:
        monkeypatch.setattr(
            probe, {"host": "host_memory", "cgroup": "cgroup_snapshot"}[stage], query_failed
        )
    with pytest.raises(RuntimeError) as caught:
        gate.instance.check("native-backward", "research")
    assert caught.value is original
    row = only_gate_row(gate)
    assert row["status"] == "UNKNOWN" and not row["admitted"]
    assert row["error"] == {"type": "RuntimeError", "message": str(original)}
    assert "cuda" not in row and row["original_cuda_free_would_refuse"] is None
    assert row["UMA_predicate_pass"] is None and not row["UMA_identity_pass"]
    assert ("observation" in row) is (stage != "host")
    assert ("cgroup" in row) is (stage == "cuda")


def test_identity_failure_keeps_valid_raw_counters_and_old_veto(probe, case, gate):
    case.identity["integrated"] = 0
    with pytest.raises(ValueError, match="must match selected Torch GB10"):
        gate.instance.check("native-backward", "research")
    row = only_gate_row(gate)
    assert row["status"] == "UNKNOWN" and not row["admitted"]
    assert row["cuda"] == case.cuda and row["observation"] == case.host
    assert row["original_cuda_free_would_refuse"] is True
    assert row["UMA_predicate_pass"] is None and not row["UMA_identity_pass"]


@pytest.mark.parametrize("failed_evidence", ["cap", "append"])
def test_query_error_survives_secondary_evidence_failure(probe, gate, monkeypatch, failed_evidence):
    original = RuntimeError("synthetic CUDA query failure")
    secondary = (
        ValueError("synthetic evidence cap")
        if failed_evidence == "cap"
        else OSError("synthetic journal fsync failure")
    )
    attempted_rows = []

    def query_failed():
        raise original

    def evidence_failed(*args):
        if failed_evidence == "append":
            attempted_rows.append(copy.deepcopy(args[1]))
        raise secondary

    monkeypatch.setattr(gate.instance, "cuda", query_failed)
    monkeypatch.setattr(
        probe, "arm_bound" if failed_evidence == "cap" else "append", evidence_failed
    )
    with pytest.raises(RuntimeError) as caught:
        gate.instance.check("native-backward", "research")
    assert caught.value is original
    assert any(
        type(secondary).__name__ in note and str(secondary) in note for note in original.__notes__
    )
    assert not gate.path.exists()
    assert len(attempted_rows) == (1 if failed_evidence == "append" else 0)
    if attempted_rows:
        assert attempted_rows[0]["status"] == "UNKNOWN"
        assert not attempted_rows[0]["admitted"]
        assert attempted_rows[0]["error"] == {
            "type": "RuntimeError",
            "message": str(original),
        }
