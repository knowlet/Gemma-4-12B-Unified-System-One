"""Synthetic stream/ownership boundaries; no sockets, model, or tokenizer IO."""
import hashlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("helper_acquisition_test_subject", HERE / "acquire.py")
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)


class Reply(io.BytesIO):
    def __init__(self, blob, *, status=200, length=None):
        super().__init__(blob)
        self.status = status
        self.headers = {} if length is None else {"Content-Length": str(length)}


def row(blob, *, lfs=False):
    value = {"rfilename": "config.json", "size": len(blob),
             "blobId": hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()}
    if lfs:
        value["lfs"] = {"sha256": hashlib.sha256(blob).hexdigest(), "size": len(blob)}
    return value


class AcquisitionBoundaries(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=subject.ROOT / "artifacts/jev-runtime")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def fetch(self, blob=b'{"synthetic":true}', **kwargs):
        selected = kwargs.pop("selected", row(blob))
        response = kwargs.pop("response", Reply(blob))
        return subject.fetch_one("config.json", selected, self.directory, 10,
                                 opener=lambda *args, **kw: response,
                                 clock=kwargs.pop("clock", lambda: 0), **kwargs)

    def test_real_pinned_metadata_identity_only(self):
        selected = subject.metadata()
        self.assertEqual(set(selected), set(subject.FILES))
        self.assertEqual(selected["model.safetensors"]["size"], 988097824)

    def test_complete_git_blob_stream_and_durable_receipt(self):
        value = self.fetch()
        self.assertTrue(value["verified"])
        self.assertEqual(value["publisher_hash_kind"], "git_blob_sha1")
        self.assertFalse((self.directory / "config.json.partial").exists())
        self.assertEqual(json.loads((self.directory / "config.json.download.json").read_text()), value)

    def test_complete_lfs_stream_uses_sha256(self):
        blob = b"independent synthetic stream"
        value = self.fetch(blob, selected=row(blob, lfs=True))
        self.assertEqual(value["publisher_hash_kind"], "lfs_sha256")

    def test_corrupt_same_size_stream_retained_without_qualified_file(self):
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            self.fetch(b"xxxx", selected=row(b"true"))
        self.assertTrue((self.directory / "config.json.partial").exists())
        self.assertFalse((self.directory / "config.json").exists())

    def test_short_stream_not_published(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.fetch(b"abc", selected=row(b"abcd"))
        self.assertFalse((self.directory / "config.json").exists())

    def test_excess_bytes_rejected_before_write(self):
        with self.assertRaisesRegex(ValueError, "byte limit"):
            self.fetch(b"abcd", selected=row(b"abc"))
        self.assertEqual((self.directory / "config.json.partial").stat().st_size, 0)

    def test_http_partial_or_wrong_length_no_file(self):
        for response in (Reply(b"abc", status=206), Reply(b"abc", length=4)):
            with self.assertRaises(ValueError):
                self.fetch(b"abc", response=response)
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_preexisting_path_preserved(self):
        (self.directory / "config.json").write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.fetch()
        self.assertEqual((self.directory / "config.json").read_bytes(), b"keep")

    def test_deadline_before_network_and_after_stream(self):
        with self.assertRaisesRegex(ValueError, "deadline"):
            self.fetch(clock=lambda: 10)
        moments = iter((0, 0, 0, 10))
        with self.assertRaisesRegex(ValueError, "late"):
            self.fetch(clock=lambda: next(moments))
        self.assertFalse((self.directory / "config.json").exists())

    def test_symlink_output_rejected(self):
        link = self.directory / "alias"
        link.symlink_to(self.directory, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            subject.ordinary(link)

    def test_failed_evidence_does_not_return_success(self):
        with patch.object(subject, "write_new", side_effect=OSError("synthetic disk failure")):
            with self.assertRaises(OSError):
                self.fetch()
        self.assertFalse((self.directory / "config.json.download.json").exists())

    def test_outside_owner_stops_before_any_output_or_network(self):
        output = self.directory / "must-not-exist"
        guard = subject.load_guard()
        with patch.object(subject, "load_guard", return_value=guard):
            with self.assertRaisesRegex(ValueError, "owner unit"):
                subject.execute(output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
