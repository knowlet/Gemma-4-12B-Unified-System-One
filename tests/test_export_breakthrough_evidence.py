"""CPU-only transport regressions for immutable Modal stage recovery."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/export_breakthrough_evidence.py"
spec = importlib.util.spec_from_file_location("export_breakthrough_evidence", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)
RUN = "20261008-breakthrough-01"
PREFIX = f"breakthrough/{RUN}/ablate"


def raw(value):
    return json.dumps(value).encode() + b"\n"


class Volume:
    def __init__(self):
        sources = {"campaign": b"# frozen campaign\n", "s1/unified.py": b"# frozen model\n"}
        receipt = {
            "status": "completed",
            "run": RUN,
            "stage": "ablate",
            "source_files": {name: export.sha256(data) for name, data in sources.items()},
            "profiles": [{"name": "released-current"}],
        }
        self.files = {
            "receipt.json": raw(receipt),
            **{f"sources/{name}.txt": data for name, data in sources.items()},
            "released-current/public231/raw/answer.json": raw({"answer": "lossless"}),
            "released-current/coherence-cache/response.json": raw({"probabilities": [0.7, 0.3]}),
            "publisher/README.md": b"license and notice\n",
        }
        self.listdir = SimpleNamespace(aio=self.list)
        self.read_file = SimpleNamespace(aio=self.read)
        self.lists, self.reads, self.active, self.peak = 0, [], 0, 0
        self.extra, self.failures, self.change = [], {}, None

    async def list(self, prefix, recursive):
        assert prefix == PREFIX and recursive is True
        self.lists += 1
        if self.lists == 2 and self.change:
            self.change(self)
        return [
            SimpleNamespace(path=PREFIX + "/" + name, type=1, size=len(data), mtime=1)
            for name, data in self.files.items()
        ] + self.extra

    async def read(self, path):
        assert path.startswith(PREFIX + "/")
        name = path.removeprefix(PREFIX + "/")
        self.reads.append(name)
        if self.failures.get(name, 0):
            self.failures[name] -= 1
            raise TimeoutError("simulated transport failure")
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            data = self.files[name]
            await asyncio.sleep(0.001)
            yield data[: len(data) // 2]
            await asyncio.sleep(0.001)
            yield data[len(data) // 2 :]
        finally:
            self.active -= 1


def recover(volume, output, **kwargs):
    return asyncio.run(export.download(volume, RUN, "ablate", output, concurrency=2, **kwargs))


def test_bounded_lossless_cpu_download_preserves_receipt_sources_and_original_failure(tmp_path):
    volume, output = Volume(), tmp_path / "recovered"
    failure = tmp_path / "original-export-timeout.json"
    failure.write_bytes(raw({"function": "export", "status": "timeout", "seconds": 600.001502}))
    result = recover(volume, output, original_failure=failure)
    assert result["verified"] is True
    assert result["gpu_receipt_status"] == "completed"
    assert result["files"] == len(volume.files)
    assert result["executed_sources_verified"] == 2
    assert volume.peak == 2 and volume.lists == 2
    for name, data in volume.files.items():
        assert (output / name).read_bytes() == data
    state = json.loads((output / export.METADATA / "receipt.json").read_text())
    assert state["settings"]["gpu_calls"] == 0
    assert state["settings"]["remote_mutations"] is False
    assert state["original_export_failure"]["sha256"] == export.sha256(failure.read_bytes())
    assert (
        output / export.METADATA / "original-export-failure.txt"
    ).read_bytes() == failure.read_bytes()
    assert not list(output.rglob(".download-*"))


def test_canonical_existing_empty_directory_requires_explicit_permission(tmp_path):
    output = tmp_path / "ablate"
    output.mkdir()
    with pytest.raises(FileExistsError):
        recover(Volume(), output)
    assert recover(Volume(), output, allow_empty_output=True)["verified"] is True
    before = (output / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        recover(Volume(), output, allow_empty_output=True)
    assert (output / "receipt.json").read_bytes() == before


@pytest.mark.parametrize(
    "entry",
    [
        (PREFIX + "/linked.json", 3, 0),
        (PREFIX + "/pipe", 4, 0),
        (PREFIX + "/../train/escaped", 1, 1),
        ("breakthrough/other/ablate/file", 1, 1),
        (PREFIX + "/C:escape", 1, 1),
        (PREFIX + "/model.safetensors", 1, 1),
        (PREFIX + "/huge.json", 1, export.MAX_FILE_BYTES + 1),
    ],
)
def test_unsafe_remote_entries_are_rejected_before_local_creation(tmp_path, entry):
    volume = Volume()
    volume.extra.append(SimpleNamespace(path=entry[0], type=entry[1], size=entry[2], mtime=1))
    output = tmp_path / "recovered"
    with pytest.raises(ValueError):
        recover(volume, output)
    assert not output.exists() and volume.reads == []


@pytest.mark.parametrize(
    "bad", ["pending_receipt", "changed_listing", "bad_source_hash", "persistent_transport"]
)
def test_partial_transport_and_remote_inconsistency_stay_failed(tmp_path, bad):
    volume, output = Volume(), tmp_path / "recovered"
    if bad == "pending_receipt":
        receipt = json.loads(volume.files["receipt.json"])
        receipt["status"] = "running"
        volume.files["receipt.json"] = raw(receipt)
    elif bad == "changed_listing":
        volume.change = lambda vol: vol.files.update({"new.json": b"new late result"})
    elif bad == "bad_source_hash":
        volume.files["sources/campaign.txt"] = b"unrecorded source"
    else:
        volume.failures["released-current/public231/raw/answer.json"] = 5
    with pytest.raises(ValueError):
        recover(volume, output, attempts=1)
    if bad == "pending_receipt":
        assert not output.exists()
    else:
        state = json.loads((output / export.METADATA / "receipt.json").read_text())
        assert state["status"] == "failed"
        assert (output / "receipt.json").read_bytes() == volume.files["receipt.json"]
        with pytest.raises(ValueError, match="incomplete"):
            export.verify_download(output)


def test_successful_retry_is_recorded_without_duplicate_or_partial_files(tmp_path):
    volume = Volume()
    member = "released-current/public231/raw/answer.json"
    volume.failures[member] = 1
    output = tmp_path / "recovered"
    assert recover(volume, output, attempts=2)["verified"] is True
    state = json.loads((output / export.METADATA / "receipt.json").read_text())
    assert next(row["attempts"] for row in state["members"] if row["path"] == member) == 2
    assert not list(output.rglob(".download-*"))


@pytest.mark.parametrize("metadata", [False, True])
def test_independent_verification_detects_download_or_metadata_corruption(tmp_path, metadata):
    output = tmp_path / "recovered"
    recover(Volume(), output)
    path = (
        output / export.METADATA / "remote-listing.json"
        if metadata
        else output / "released-current/coherence-cache/response.json"
    )
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash/size"):
        export.verify_download(output)


def test_post_listing_same_size_receipt_mutation_is_detected(tmp_path):
    volume, output = Volume(), tmp_path / "recovered"
    original = volume.files["receipt.json"]
    volume.change = lambda vol: vol.files.update(
        {"receipt.json": original.replace(b"released-current", b"different-result")}
    )
    assert len(original) == len(original.replace(b"released-current", b"different-result"))
    with pytest.raises(ValueError, match="changed"):
        recover(volume, output)


def test_local_output_symlink_is_rejected_before_any_remote_read(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "linked"
    output.symlink_to(target, target_is_directory=True)
    volume = Volume()
    with pytest.raises(ValueError, match="symlink"):
        recover(volume, output, allow_empty_output=True)
    assert volume.lists == 0 and not list(target.iterdir())


def test_train_plan_includes_head_features_and_adapter_but_never_baseline_weights():
    prefix = f"breakthrough/{RUN}/train"
    names = [
        "receipt.json",
        "decision-head.pt",
        "train-features.pt",
        "mixed-lora/adapter/adapter_model.safetensors",
    ]
    entries = [SimpleNamespace(path=prefix + "/" + name, type=1, size=1, mtime=1) for name in names]
    assert set(export._plan(entries, prefix, "train")) == set(names)
    entries.append(SimpleNamespace(path=prefix + "/model.safetensors", type=1, size=1, mtime=1))
    with pytest.raises(ValueError, match="weights"):
        export._plan(entries, prefix, "train")


def test_unexpected_local_file_during_transfer_is_retained_and_transfer_stays_failed(tmp_path):
    volume, output = Volume(), tmp_path / "recovered"
    volume.change = lambda vol: (output / "unrelated.txt").write_bytes(b"keep other writer's data")
    with pytest.raises(ValueError, match="population"):
        recover(volume, output)
    assert (output / "unrelated.txt").read_bytes() == b"keep other writer's data"
    state = json.loads((output / export.METADATA / "receipt.json").read_text())
    assert state["status"] == "failed"


def test_failed_gpu_stage_can_be_archived_without_relabeling_it_completed(tmp_path):
    volume, output = Volume(), tmp_path / "recovered"
    receipt = json.loads(volume.files["receipt.json"])
    receipt["status"] = "failed"
    receipt["error"] = "original GPU failure"
    volume.files["receipt.json"] = raw(receipt)
    result = recover(volume, output)
    assert result["download_status"] == "completed"
    assert result["gpu_receipt_status"] == "failed"
    assert (output / "receipt.json").read_bytes() == volume.files["receipt.json"]


def test_declared_maximum_32_concurrency_preserves_every_byte_and_bounded_io(tmp_path):
    volume, output = Volume(), tmp_path / "recovered"
    volume.files.update({f"raw/case-{index:03}.json": raw({"case": index}) for index in range(96)})
    result = asyncio.run(export.download(volume, RUN, "ablate", output, concurrency=32))
    assert volume.peak == 32
    assert result["verified"] is True and result["files"] == len(volume.files)
    for name, data in volume.files.items():
        assert (output / name).read_bytes() == data
    state = json.loads((output / export.METADATA / "receipt.json").read_text())
    assert state["settings"]["concurrency"] == 32
    assert state["settings"]["remote_mutations"] is False
    assert all(member["attempts"] == 1 for member in state["members"])
