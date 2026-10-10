"""In-memory ZIP and local fixture checks; no cloud calls or model loading."""

import copy
import hashlib
import importlib.util
import io
import json
import stat
import struct
import sys
import types
import zipfile
from contextlib import nullcontext
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


RUN = "20261008-breakthrough-01"
SOURCE_SHA = identity(b"fixture verifier source")["sha256"]


@pytest.fixture
def large_result():
    return bytes(range(256)) * (3 * 1024 * 1024 // 256) + b"complete result tail"


def test_large_result_is_exact_with_bounded_byte_chunks_and_terminal_digest(large_result):
    records = list(audit.result_stream(large_result, RUN, SOURCE_SHA))
    assert len(large_result) > 2 * 1024 * 1024
    assert audit.assemble_result_stream(records, RUN, SOURCE_SHA) == large_result
    assert all(len(r["data"]) <= 256 * 1024 for r in records[1:-1])
    assert records[-1]["sha256"] == identity(large_result)["sha256"]


def test_each_optional_modal_serialized_yield_avoids_blob_upload(large_result):
    serialize = pytest.importorskip("modal._serialization").serialize
    maximum = pytest.importorskip("modal._utils.blob_utils").MAX_OBJECT_SIZE_BYTES
    records = list(audit.result_stream(large_result, RUN, SOURCE_SHA))
    assert len(serialize(large_result)) > maximum
    for record in records:
        assert len(serialize(record)) <= audit.RESULT_CHUNK_BYTES + 1024
        assert len(serialize(record)) < maximum


@pytest.mark.parametrize("size", [1, 256 * 1024 - 1, 256 * 1024, 256 * 1024 + 1])
def test_result_stream_chunk_boundary_sizes_are_exact(size):
    raw = b"X" * size
    records = list(audit.result_stream(raw, RUN, SOURCE_SHA))
    assert (
        records[0]["chunk_count"]
        == (size + audit.RESULT_CHUNK_BYTES - 1) // audit.RESULT_CHUNK_BYTES
    )
    assert audit.assemble_result_stream(records, RUN, SOURCE_SHA) == raw


@pytest.mark.parametrize("bad", [b"", "not bytes", bytearray(b"bytes")])
def test_result_stream_requires_bounded_nonempty_bytes(bad):
    with pytest.raises(ValueError):
        list(audit.result_stream(bad, RUN, SOURCE_SHA))


def test_result_stream_rejects_result_total_bound_and_accepts_exact_bound(monkeypatch):
    monkeypatch.setattr(audit, "MAX_RESULT_BYTES", 8)
    records = list(audit.result_stream(b"exactly8", RUN, SOURCE_SHA))
    assert audit.assemble_result_stream(records, RUN, SOURCE_SHA) == b"exactly8"
    with pytest.raises(ValueError):
        list(audit.result_stream(b"too large", RUN, SOURCE_SHA))
    monkeypatch.setattr(audit, "MAX_RESULT_BYTES", 7)
    with pytest.raises(ValueError):
        audit.assemble_result_stream(records, RUN, SOURCE_SHA)


def test_remote_wrapper_passes_every_audit_argument_unchanged(monkeypatch):
    args = (RUN, {"input": "identity"}, SOURCE_SHA, b"input bytes", {"origin": "identity"}, True)
    calls = []

    def original(*values):
        calls.append(values)
        return b"exact original result bytes"

    monkeypatch.setattr(audit, "audit_run", original)
    records = list(audit.audit_run_stream(*args))
    assert calls == [args]
    assert audit.assemble_result_stream(records, RUN, SOURCE_SHA) == b"exact original result bytes"


@pytest.mark.parametrize(
    "bad",
    [
        "missing_header",
        "missing_chunk",
        "duplicate_chunk",
        "out_of_order",
        "missing_terminal",
        "extra_chunk",
        "extra_after_terminal",
        "extra_header_field",
        "wrong_protocol",
        "wrong_run",
        "wrong_source",
        "header_count",
        "header_count_bool",
        "header_size",
        "header_size_bool",
        "header_chunk_bytes",
        "header_chunk_bytes_bool",
        "header_hash",
        "header_hash_type",
        "chunk_index",
        "chunk_index_bool",
        "chunk_size",
        "chunk_size_bool",
        "chunk_hash",
        "chunk_data",
        "chunk_data_type",
        "chunk_extra",
        "terminal_hash",
        "terminal_count",
        "terminal_count_bool",
        "terminal_size",
        "terminal_size_bool",
        "terminal_extra",
        "changed_whole_hash",
        "header_not_dict",
        "chunk_not_dict",
        "terminal_not_dict",
    ],
)
def test_result_assembler_rejects_partial_malicious_or_ambiguous_stream(large_result, bad):
    records = list(audit.result_stream(large_result, RUN, SOURCE_SHA))
    if bad == "missing_header":
        records.pop(0)
    elif bad == "missing_chunk":
        records.pop(2)
    elif bad == "duplicate_chunk":
        records.insert(2, copy.deepcopy(records[1]))
    elif bad == "out_of_order":
        records[1], records[2] = records[2], records[1]
    elif bad == "missing_terminal":
        records.pop()
    elif bad == "extra_chunk":
        records.insert(-1, copy.deepcopy(records[1]))
    elif bad == "extra_after_terminal":
        records.append(copy.deepcopy(records[-1]))
    elif bad == "extra_header_field":
        records[0]["extra"] = "unexpected"
    elif bad in {"wrong_protocol", "wrong_run", "wrong_source"}:
        field = {
            "wrong_protocol": "protocol",
            "wrong_run": "run",
            "wrong_source": "verifier_source_sha256",
        }[bad]
        records[0][field] = "other identity"
    elif bad.startswith("header_"):
        field = {
            "header_count": "chunk_count",
            "header_count_bool": "chunk_count",
            "header_size": "size_bytes",
            "header_size_bool": "size_bytes",
            "header_chunk_bytes": "chunk_bytes",
            "header_chunk_bytes_bool": "chunk_bytes",
            "header_hash": "sha256",
            "header_hash_type": "sha256",
        }.get(bad)
        if bad == "header_not_dict":
            records[0] = []
        else:
            records[0][field] = (
                True if bad.endswith("bool") else (None if bad.endswith("type") else -1)
            )
    elif bad.startswith("chunk_"):
        if bad == "chunk_extra":
            records[1]["extra"] = "unexpected"
        elif bad == "chunk_data":
            records[1]["data"] = b"changed bytes"
        elif bad == "chunk_data_type":
            records[1]["data"] = bytearray(records[1]["data"])
        elif bad == "chunk_not_dict":
            records[1] = []
        else:
            field = {
                "chunk_index": "index",
                "chunk_index_bool": "index",
                "chunk_size": "size_bytes",
                "chunk_size_bool": "size_bytes",
                "chunk_hash": "sha256",
            }[bad]
            records[1][field] = True if bad.endswith("bool") else -1
    elif bad == "changed_whole_hash":
        records[0]["sha256"] = records[-1]["sha256"] = identity(b"other complete bytes")["sha256"]
    elif bad == "terminal_not_dict":
        records[-1] = []
    elif bad == "terminal_extra":
        records[-1]["extra"] = "unexpected"
    else:
        field = {
            "terminal_hash": "sha256",
            "terminal_count": "chunk_count",
            "terminal_count_bool": "chunk_count",
            "terminal_size": "size_bytes",
            "terminal_size_bool": "size_bytes",
        }[bad]
        records[-1][field] = True if bad.endswith("bool") else -1
    with pytest.raises(ValueError):
        audit.assemble_result_stream(records, RUN, SOURCE_SHA)


@pytest.mark.parametrize("cut", [0, 1, 2, -1])
def test_truncated_stream_retains_only_verified_bytes_and_failure_receipt(
    tmp_path, large_result, cut
):
    records = list(audit.result_stream(large_result, RUN, SOURCE_SHA))[:cut]
    partial, receipt = tmp_path / "result.zip.partial", tmp_path / "result-stream.json"
    with pytest.raises(ValueError, match="truncated"):
        audit.assemble_result_stream(
            records, RUN, SOURCE_SHA, partial_path=partial, receipt_path=receipt
        )
    retained = partial.read_bytes()
    proof = json.loads(receipt.read_bytes())
    assert proof["status"] == "failed"
    assert proof["error_type"] == "ValueError"
    assert proof["received_size_bytes"] == len(retained)
    assert proof["received_sha256"] == identity(retained)["sha256"]
    assert len(retained) == sum(r["size_bytes"] for r in records if r["kind"] == "chunk")
    assert "data" not in str(proof["chunks"])


@pytest.mark.parametrize("after_terminal", [False, True])
def test_remote_error_is_propagated_with_partial_transport_evidence(
    tmp_path, large_result, after_terminal
):
    records = list(audit.result_stream(large_result, RUN, SOURCE_SHA))

    def failed():
        yield from records if after_terminal else records[:2]
        raise RuntimeError("remote generator delivery failed")

    partial, receipt = tmp_path / "result.zip.partial", tmp_path / "result-stream.json"
    with pytest.raises(RuntimeError, match="remote generator delivery failed"):
        audit.assemble_result_stream(
            failed(), RUN, SOURCE_SHA, partial_path=partial, receipt_path=receipt
        )
    proof = json.loads(receipt.read_bytes())
    assert proof["status"] == "failed"
    assert proof["error_type"] == "RuntimeError"
    assert proof["error"] == "remote generator delivery failed"
    assert partial.read_bytes() == (
        large_result if after_terminal else large_result[: audit.RESULT_CHUNK_BYTES]
    )


def test_completed_stream_saves_all_bound_transport_metadata(tmp_path, large_result):
    partial, receipt = tmp_path / "result.zip.partial", tmp_path / "result-stream.json"
    records = list(audit.result_stream(large_result, RUN, SOURCE_SHA))
    result = audit.assemble_result_stream(
        records, RUN, SOURCE_SHA, partial_path=partial, receipt_path=receipt
    )
    assert result == partial.read_bytes() == large_result
    proof = json.loads(receipt.read_bytes())
    assert proof["status"] == "completed"
    assert proof["header"] == records[0]
    assert proof["terminal"] == records[-1]
    assert proof["chunks"] == [{k: v for k, v in r.items() if k != "data"} for r in records[1:-1]]
    assert proof["received_sha256"] == identity(large_result)["sha256"]


@pytest.mark.parametrize("occupied", ["partial", "receipt"])
def test_transport_evidence_paths_are_exclusive_and_existing_bytes_survive(tmp_path, occupied):
    partial, receipt = tmp_path / "result.zip.partial", tmp_path / "result-stream.json"
    existing = partial if occupied == "partial" else receipt
    existing.write_bytes(b"prior evidence")
    with pytest.raises(ValueError, match="exclusive new"):
        audit.assemble_result_stream(
            [], RUN, SOURCE_SHA, partial_path=partial, receipt_path=receipt
        )
    assert existing.read_bytes() == b"prior evidence"
    assert not (receipt if occupied == "partial" else partial).exists()


def test_mutated_previous_header_cannot_change_the_validated_total_or_hash(large_result):
    records = list(audit.result_stream(large_result, RUN, SOURCE_SHA))

    def malicious():
        yield records[0]
        records[0]["size_bytes"] = 1
        records[0]["sha256"] = identity(b"X")["sha256"]
        yield from records[1:]

    assert audit.assemble_result_stream(malicious(), RUN, SOURCE_SHA) == large_result


@pytest.mark.parametrize("failed", [False, True])
def test_main_uses_generator_and_preserves_all_pinned_resource_guards(
    tmp_path, roots, files, monkeypatch, failed
):
    observed = {}
    source = SCRIPT.read_bytes()
    proof = b'{"status":"completed","padding":"' + b"X" * (3 * 1024 * 1024) + b'"}'
    result = make_zip([("proof.json", proof), ("verifier.py.txt", source)])
    assert len(result) > 2 * 1024 * 1024

    def original(*args):
        observed["args"] = args
        return result

    class Remote:
        object_id = "fake-function"

        def __init__(self, function):
            self.function = function

        def remote_gen(self, *args):
            observed["method"] = "remote_gen"
            records = self.function(*args)
            if failed:
                yield next(records)
                yield next(records)
                raise RuntimeError("retained delivery failure")
            yield from records

    class App:
        app_id = "fake-app"

        def __init__(self, name):
            observed["name"] = name

        def function(self, **kwargs):
            observed["resources"] = kwargs
            return Remote

        def run(self):
            return nullcontext()

    class Volume:
        @classmethod
        def from_name(cls, name, create_if_missing):
            observed["volume"] = (name, create_if_missing)
            return cls()

        def with_mount_options(self, **options):
            observed["mount_options"] = options
            return self

    fake_modal = types.SimpleNamespace(
        App=App, Image=types.SimpleNamespace(from_id=lambda x: x), Volume=Volume
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(audit, "audit_run", original)
    monkeypatch.setattr(audit, "collect_input_manifest", lambda *args, **kwargs: manifest(files))
    monkeypatch.setattr(audit, "snapshot_origin", lambda *args: {"fixture": "exact origin"})
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--run",
            RUN,
            "--ablate",
            str(roots["ablate"]),
            "--train",
            str(roots["train"]),
            "--output",
            str(output),
        ],
    )
    if failed:
        with pytest.raises(RuntimeError, match="retained delivery failure"):
            audit.main()
        assert not (output / "result.zip").exists()
        assert not (output / "proof.json").exists()
        assert (output / "result.zip.partial").read_bytes() == result[: audit.RESULT_CHUNK_BYTES]
        assert (
            json.loads((output / "failure.json").read_bytes())["error"]
            == "retained delivery failure"
        )
        assert json.loads((output / "result-stream.json").read_bytes())["status"] == "failed"
    else:
        audit.main()
        assert (output / "result.zip").read_bytes() == result
        assert (output / "proof.json").read_bytes() == proof
        assert not (output / "result.zip.partial").exists()
        assert json.loads((output / "result-stream.json").read_bytes())["status"] == "completed"
    assert observed["method"] == "remote_gen"
    assert observed["args"] == (
        RUN,
        manifest(files),
        identity(source)["sha256"],
        (output / "input.zip").read_bytes(),
        {"fixture": "exact origin"},
        False,
    )
    assert observed["volume"] == (audit.VOLUME, False)
    assert observed["mount_options"] == {"read_only": True}
    assert observed["resources"] == {
        "image": audit.IMAGE_ID,
        "cpu": 2,
        "memory": 4096,
        "timeout": 600,
        "volumes": {"/vol": observed["resources"]["volumes"]["/vol"]},
        "block_network": True,
        "max_containers": 1,
        "single_use_containers": True,
    }
