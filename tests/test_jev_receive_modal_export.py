"""Parallel local receive integrity/cancellation; all Volume reads are fake."""

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/jev_receive_modal_export.py"
SPEC = importlib.util.spec_from_file_location("receive_modal_export_test_subject", PATH)
RECEIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECEIVER)
ROOT = "public07-h200-120-20261018-09-fit"


def pin(blob):
    return {"sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}


class Volume:
    def __init__(self, files, *, index=None):
        self.payloads = files
        self.index = (
            index
            or json.dumps(
                {
                    "root": ROOT,
                    "complete_owned_exit": True,
                    "files": {name: pin(blob) for name, blob in files.items()},
                }
            ).encode()
        )
        self.active, self.peak, self.calls = 0, 0, []
        self.finish = lambda name: None
        self.read_file = SimpleNamespace(aio=self.read)

    async def read(self, path):
        name = path.removeprefix(ROOT + "/")
        assert path == ROOT + "/" + name
        self.calls.append(name)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            blob = self.index if name == "export-index.json" else self.payloads[name]
            for position in range(0, len(blob), 3):
                await asyncio.sleep(0)
                yield blob[position : position + 3]
        finally:
            self.active -= 1
            self.finish(name)


def receive(volume, output, **kwargs):
    return asyncio.run(
        RECEIVER.receive(
            volume,
            remote_root=ROOT,
            index_sha256=pin(volume.index)["sha256"],
            index_size_bytes=len(volume.index),
            output=output,
            **kwargs,
        )
    )


def test_parallel_streams_verify_every_byte_and_exact_tree_receipt_is_outside(tmp_path):
    payloads = {f"nested/{i}.json": bytes([i]) * 50 for i in range(20)}
    payloads.update(
        {"empty": b"", ".pending-row": b"partial evidence", "unicode/檔案": b"utf8 name"}
    )
    volume, output = Volume(payloads), tmp_path / "new"
    result = receive(volume, output)
    assert result["status"] == "complete" and result["complete_owned_exit"] is True
    assert 1 < volume.peak <= 8 and volume.active == 0
    assert set(volume.calls) == {*payloads, "export-index.json"}
    assert json.loads((output / "transfer-receipt.json").read_bytes()) == result
    assert (output / "tree/export-index.json").read_bytes() == volume.index
    assert all((output / "tree" / name).read_bytes() == blob for name, blob in payloads.items())
    assert (
        len(list(path for path in (output / "tree").rglob("*") if path.is_file()))
        == len(payloads) + 1
    )
    with pytest.raises(ValueError, match="Fresh"):
        receive(volume, output)


@pytest.mark.parametrize(
    "fault",
    ["duplicate", "unclosed", "traversal", "prefix", "self", "bool_size", "file_cap", "byte_cap"],
)
def test_invalid_closed_index_never_starts_payload_reads_and_retains_failure(tmp_path, fault):
    index = {"root": ROOT, "complete_owned_exit": True, "files": {"file": pin(b"abc")}}
    if fault == "duplicate":
        blob = (
            b'{"root":"'
            + ROOT.encode()
            + b'","root":"duplicate","complete_owned_exit":true,"files":{}}'
        )
    else:
        if fault == "unclosed":
            index["complete_owned_exit"] = False
        elif fault == "traversal":
            index["files"] = {"../outside": pin(b"abc")}
        elif fault == "prefix":
            index["files"]["file/child"] = pin(b"abc")
        elif fault == "self":
            index["files"]["export-index.json"] = pin(b"abc")
        elif fault == "bool_size":
            index["files"]["file"]["size_bytes"] = True
        elif fault == "file_cap":
            index["files"] = {str(i): pin(b"") for i in range(RECEIVER.MAX_FILES)}
        else:
            index["files"]["file"]["size_bytes"] = RECEIVER.MAX_BYTES
        blob = json.dumps(index).encode()
    volume, output = Volume({}, index=blob), tmp_path / "new"
    with pytest.raises(ValueError):
        receive(volume, output)
    assert volume.calls == ["export-index.json"] and volume.active == 0
    report = json.loads((output / "transfer-receipt.json").read_bytes())
    assert report["status"] == "failed" and "complete_owned_exit" not in report
    assert (output / "tree/export-index.json").read_bytes() == blob


def test_corrupt_stream_cancels_and_awaits_sibling_and_retains_partial_bytes(tmp_path):
    volume = Volume({"fail": b"abc", "slow": b"xyz"})
    started = asyncio.Event()
    active, closed = set(), set()

    async def read(path):
        name = path.removeprefix(ROOT + "/")
        if name == "export-index.json":
            yield volume.index
            return
        active.add(name)
        try:
            if name == "slow":
                started.set()
                yield b"x"
                await asyncio.Event().wait()
            else:
                await started.wait()
                yield b"bad"
        finally:
            active.remove(name)
            closed.add(name)

    volume.read_file.aio = read
    output = tmp_path / "new"
    with pytest.raises(ExceptionGroup):
        receive(volume, output)
    assert not active and closed == {"fail", "slow"}
    assert (output / "tree/slow").read_bytes() == b"x"
    assert json.loads((output / "transfer-receipt.json").read_bytes())["status"] == "failed"


def test_external_cancellation_closes_async_iterator_before_failure_receipt(tmp_path):
    volume, output = Volume({"file": b"abc"}), tmp_path / "new"
    started, closed = asyncio.Event(), []

    async def read(path):
        if path.endswith("export-index.json"):
            yield volume.index
            return
        try:
            yield b"a"
            started.set()
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    volume.read_file.aio = read

    async def scenario():
        task = asyncio.create_task(
            RECEIVER.receive(
                volume,
                remote_root=ROOT,
                index_sha256=pin(volume.index)["sha256"],
                index_size_bytes=len(volume.index),
                output=output,
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert closed == [True] and (output / "tree/file").read_bytes() == b"a"
    assert (
        json.loads((output / "transfer-receipt.json").read_bytes())["error"]["type"]
        == "CancelledError"
    )


@pytest.mark.parametrize("fault", ["unknown_file", "unknown_directory", "symlink"])
def test_unexpected_tree_entries_or_links_refuse_completion_without_overwrite(tmp_path, fault):
    output, outside = tmp_path / "new", tmp_path / "outside"
    outside.write_bytes(b"do not overwrite")
    volume = Volume({"file": b"abc"})

    def finish(name):
        if name == "file":
            path = output / "tree/unknown"
            if fault == "symlink":
                path.symlink_to(outside)
            elif fault == "unknown_directory":
                path.mkdir()
            else:
                path.write_bytes(b"unknown")

    volume.finish = finish
    with pytest.raises(ValueError):
        receive(volume, output)
    assert outside.read_bytes() == b"do not overwrite"
    assert json.loads((output / "transfer-receipt.json").read_bytes())["status"] == "failed"


def test_external_index_pin_and_symlink_ancestor_refuse_before_payloads(tmp_path):
    volume = Volume({"file": b"abc"})
    with pytest.raises(ValueError, match="SHA/size"):
        asyncio.run(
            RECEIVER.receive(
                volume,
                remote_root=ROOT,
                index_sha256="0" * 64,
                index_size_bytes=len(volume.index),
                output=tmp_path / "bad-pin",
            )
        )
    assert volume.calls == ["export-index.json"]
    storage = tmp_path / "storage"
    storage.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(storage, target_is_directory=True)
    with pytest.raises(ValueError, match="Ordinary"):
        receive(volume, alias / "new")
    assert not list(storage.iterdir())
