"""In-memory ZIP and local fixture checks; no cloud calls or model loading."""

import hashlib
import importlib.util
import io
import json
import stat
import struct
import zipfile
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/audit_breakthrough_arithmetic_zip.py"
SPEC = importlib.util.spec_from_file_location("audit_breakthrough_arithmetic_zip", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def identity(raw):
    return {"size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def manifest(files):
    return {name: identity(raw) for name, raw in files.items()}


def make_zip(entries, *, compression=zipfile.ZIP_STORED, mode="regular"):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, raw in entries:
            info = zipfile.ZipInfo(name, date_time=FIXED_TIMESTAMP)
            info.compress_type = compression
            info.create_system = 3 if mode != "unspecified" else 0
            if mode == "regular":
                info.external_attr = (stat.S_IFREG | 0o644) << 16
            elif mode == "symlink":
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
            elif mode == "directory":
                info.external_attr = (stat.S_IFDIR | 0o755) << 16
            archive.writestr(info, raw)
    return stream.getvalue()


@pytest.fixture
def files():
    return {
        "train/sources/example.py.txt": b"# exact source\n\x00\xff",
        "ablate/receipt.json": b'{"stage":"ablate","status":"completed"}\n',
    }


@pytest.fixture
def roots(tmp_path, files):
    roots = {stage: tmp_path / stage for stage in ("ablate", "train")}
    for name, raw in files.items():
        stage, relative = name.split("/", 1)
        path = roots[stage] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    return roots


@pytest.mark.parametrize("mode", ["regular", "unspecified"])
@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_zip_returns_exact_two_member_bytes_with_verified_size_and_sha(files, mode, compression):
    payload = make_zip(files.items(), mode=mode, compression=compression)
    result = audit.validate_input_zip(payload, manifest(files))
    assert result == files
    assert {name: identity(raw) for name, raw in result.items()} == manifest(files)


@pytest.mark.parametrize("bad", ["missing", "extra"])
def test_zip_rejects_different_member_population(files, bad):
    entries = list(files.items())
    if bad == "missing":
        entries.pop()
    else:
        entries.append(("ablate/extra.json", b"{}\n"))
    with pytest.raises(ValueError):
        audit.validate_input_zip(make_zip(entries), manifest(files))


def test_zip_rejects_duplicate_member_even_when_population_and_bytes_match(files):
    entries = [*files.items(), next(iter(files.items()))]
    with pytest.warns(UserWarning, match="Duplicate name"):
        payload = make_zip(entries)
    with pytest.raises(ValueError):
        audit.validate_input_zip(payload, manifest(files))


@pytest.mark.parametrize(
    "name",
    [
        "../ablate/receipt.json",
        "ablate/../train/receipt.json",
        "/ablate/receipt.json",
        "ablate//receipt.json",
        "ablate\\receipt.json",
        "ablate/C:receipt.json",
    ],
)
def test_zip_rejects_unsafe_member_even_if_declared_in_expected_population(name):
    files = {name: b"source bytes"}
    with pytest.raises(ValueError):
        audit.validate_input_zip(make_zip(files.items()), manifest(files))


@pytest.mark.parametrize("mode", ["symlink", "directory"])
def test_zip_rejects_nonregular_unix_member_mode(files, mode):
    with pytest.raises(ValueError):
        audit.validate_input_zip(make_zip(files.items(), mode=mode), manifest(files))


def test_zip_rejects_directory_entry_name(files):
    entries = [*files.items(), ("ablate/sources/", b"")]
    expected = {**manifest(files), "ablate/sources/": identity(b"")}
    with pytest.raises(ValueError):
        audit.validate_input_zip(make_zip(entries), expected)


def test_zip_rejects_encrypted_member_flag_before_extracting(files):
    payload = bytearray(make_zip(files.items()))
    local, central = payload.index(b"PK\x03\x04"), payload.index(b"PK\x01\x02")
    for offset in (local + 6, central + 8):
        flags = struct.unpack_from("<H", payload, offset)[0]
        struct.pack_into("<H", payload, offset, flags | 1)
    with pytest.raises(ValueError):
        audit.validate_input_zip(bytes(payload), manifest(files))


@pytest.mark.parametrize("compression", [zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA])
def test_zip_rejects_unsupported_compression(files, compression):
    with pytest.raises(ValueError):
        audit.validate_input_zip(make_zip(files.items(), compression=compression), manifest(files))


@pytest.mark.parametrize("field", ["size_bytes", "sha256"])
def test_zip_rejects_member_with_wrong_expected_identity(files, field):
    expected = manifest(files)
    name = next(iter(files))
    expected[name][field] = (
        len(files[name]) + 1 if field == "size_bytes" else identity(b"other bytes")["sha256"]
    )
    with pytest.raises(ValueError):
        audit.validate_input_zip(make_zip(files.items()), expected)


@pytest.mark.parametrize("bound", ["MAX_MEMBER_BYTES", "MAX_TOTAL_BYTES", "MAX_ZIP_BYTES"])
def test_zip_rejects_exceeding_each_bound_with_otherwise_valid_payload(files, monkeypatch, bound):
    payload = make_zip(files.items())
    measured = {
        "MAX_MEMBER_BYTES": max(map(len, files.values())),
        "MAX_TOTAL_BYTES": sum(map(len, files.values())),
        "MAX_ZIP_BYTES": len(payload),
    }[bound]
    monkeypatch.setattr(audit, bound, measured - 1)
    with pytest.raises(ValueError):
        audit.validate_input_zip(payload, manifest(files))


def test_zip_accepts_exact_inclusive_member_total_and_compressed_bounds(files, monkeypatch):
    payload = make_zip(files.items())
    monkeypatch.setattr(audit, "MAX_MEMBER_BYTES", max(map(len, files.values())))
    monkeypatch.setattr(audit, "MAX_TOTAL_BYTES", sum(map(len, files.values())))
    monkeypatch.setattr(audit, "MAX_ZIP_BYTES", len(payload))
    assert audit.validate_input_zip(payload, manifest(files)) == files


def test_pack_is_deterministic_sorted_fixed_timestamp_and_lossless(roots, files):
    expected = manifest(files)
    first = audit.pack_input_zip(roots, expected)
    second = audit.pack_input_zip(
        dict(reversed(list(roots.items()))), dict(reversed(list(expected.items())))
    )
    assert first == second
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.namelist() == sorted(expected)
        assert all(info.date_time == FIXED_TIMESTAMP for info in archive.infolist())
        assert {name: archive.read(name) for name in archive.namelist()} == files
    assert audit.validate_input_zip(first, expected) == files


@pytest.mark.parametrize("mutation", ["same_size_wrong_sha", "wrong_size"])
def test_pack_verifies_actual_root_file_bytes_against_manifest(roots, files, mutation):
    expected = manifest(files)
    name = next(iter(files))
    stage, relative = name.split("/", 1)
    changed = b"X" * len(files[name]) if mutation == "same_size_wrong_sha" else files[name] + b"X"
    (roots[stage] / relative).write_bytes(changed)
    with pytest.raises(ValueError):
        audit.pack_input_zip(roots, expected)


@pytest.mark.parametrize("bound", ["MAX_MEMBER_BYTES", "MAX_TOTAL_BYTES", "MAX_ZIP_BYTES"])
def test_pack_enforces_the_same_input_bounds(roots, files, monkeypatch, bound):
    expected = manifest(files)
    normal_payload = audit.pack_input_zip(roots, expected)
    measured = {
        "MAX_MEMBER_BYTES": max(map(len, files.values())),
        "MAX_TOTAL_BYTES": sum(map(len, files.values())),
        "MAX_ZIP_BYTES": len(normal_payload),
    }[bound]
    monkeypatch.setattr(audit, bound, measured - 1)
    with pytest.raises(ValueError):
        audit.pack_input_zip(roots, expected)


def write_origin(directory, receipt):
    raw = (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()
    path = directory / ".evidence-download/receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


@pytest.fixture
def origin_snapshot(tmp_path):
    run = "20261008-breakthrough-01"
    roots, expected, receipts = {}, {}, {}
    for stage in ("ablate", "train"):
        roots[stage] = tmp_path / stage
        source = b"# exact executed source\n"
        gpu = (
            json.dumps(
                {
                    "run": run,
                    "stage": stage,
                    "status": "completed",
                    "source_files": {"example.py": identity(source)["sha256"]},
                },
                sort_keys=True,
            )
            + "\n"
        ).encode()
        files = {
            "receipt.json": gpu,
            "sources/example.py.txt": source,
            "unused-result.json": b'{"not_used_for_arithmetic":true}\n',
        }
        for relative, raw in files.items():
            path = roots[stage] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            if relative != "unused-result.json":
                expected[stage + "/" + relative] = identity(raw)
        receipt = {
            "schema_version": 1,
            "status": "completed",
            "errors": [],
            "run": run,
            "stage": stage,
            "volume": "gemma-unified-system-one",
            "remote_prefix": "breakthrough/" + run + "/" + stage,
            "gpu_receipt_sha256": identity(gpu)["sha256"],
            "source_sha256": identity(b"# transfer helper fixture\n")["sha256"],
            "expected_files": len(files),
            "expected_size_bytes": sum(map(len, files.values())),
            "members": [
                {"path": relative, **identity(raw)} for relative, raw in sorted(files.items())
            ],
        }
        write_origin(roots[stage], receipt)
        receipts[stage] = receipt
    return roots, expected, run, receipts


def test_snapshot_origin_proves_exact_used_bytes_from_complete_stage_transfers(origin_snapshot):
    roots, expected, run, receipts = origin_snapshot
    assert audit.snapshot_origin(roots, expected, run) == {
        stage: {
            "run": run,
            "stage": stage,
            "volume": "gemma-unified-system-one",
            "remote_prefix": "breakthrough/" + run + "/" + stage,
            "transfer_receipt_sha256": identity(
                (root / ".evidence-download/receipt.json").read_bytes()
            )["sha256"],
            "terminal_gpu_receipt_sha256": expected[stage + "/receipt.json"]["sha256"],
            "transfer_source_sha256": receipts[stage]["source_sha256"],
            "transfer_files": 3,
            "transfer_size_bytes": receipts[stage]["expected_size_bytes"],
        }
        for stage, root in roots.items()
    }


@pytest.mark.parametrize(
    "bad",
    [
        "run",
        "stage",
        "remote_prefix",
        "volume",
        "gpu_receipt_sha256",
        "member_sha",
        "member_size",
        "status",
        "errors",
        "expected_files",
        "expected_size_bytes",
        "missing_member",
        "duplicate_member",
    ],
)
def test_snapshot_origin_rejects_wrong_identity_or_incomplete_transfer(origin_snapshot, bad):
    roots, expected, run, receipts = origin_snapshot
    receipt = receipts["train"]
    if bad in {"run", "stage", "remote_prefix", "volume"}:
        receipt[bad] = "other"
    elif bad == "gpu_receipt_sha256":
        receipt[bad] = identity(b"other terminal GPU receipt")["sha256"]
    elif bad in {"member_sha", "member_size"}:
        member = next(row for row in receipt["members"] if row["path"] == "sources/example.py.txt")
        if bad == "member_sha":
            member["sha256"] = identity(b"other executed source")["sha256"]
        else:
            member["size_bytes"] += 1
    elif bad == "status":
        receipt[bad] = "running"
    elif bad == "errors":
        receipt[bad] = [{"path": "sources/example.py.txt", "error_type": "TimeoutError"}]
    elif bad in {"expected_files", "expected_size_bytes"}:
        receipt[bad] += 1
    elif bad == "missing_member":
        receipt["members"].pop()
    else:
        receipt["members"][-1]["path"] = receipt["members"][0]["path"]
    write_origin(roots["train"], receipt)
    with pytest.raises(ValueError):
        audit.snapshot_origin(roots, expected, run)


@pytest.mark.parametrize("bad", ["prepended", "trailing", "concatenated", "comment"])
def test_zip_rejects_ambiguous_container_even_with_exact_members(files, bad):
    payload = make_zip(files.items())
    if bad == "prepended":
        payload = b"unrelated prefix" + payload
    elif bad == "trailing":
        payload += b"unrelated suffix"
    elif bad == "concatenated":
        payload += payload
    else:
        buffer = io.BytesIO(payload)
        with zipfile.ZipFile(buffer, "a") as archive:
            archive.comment = b"comment bytes"
        payload = buffer.getvalue()
    with pytest.raises(ValueError, match="unambiguous single ZIP container"):
        audit.validate_input_zip(payload, manifest(files))
