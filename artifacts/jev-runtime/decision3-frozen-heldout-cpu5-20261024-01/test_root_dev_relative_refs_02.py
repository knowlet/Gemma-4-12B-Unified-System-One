"""Regression for genuine repo-relative root proof refs; invented local JSON only."""

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path


class RelativeProofTests(unittest.TestCase):
    def test_selected_root_relative_proof_authenticates_and_corruption_refuses(self):
        h = Path(__file__).resolve().parent
        p = h / "root-closed-dev-chunk-audit-02.py"
        s = importlib.util.spec_from_file_location("_dev_audit_v2_fake", p)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        with tempfile.TemporaryDirectory(prefix="invented-proof-", dir=h) as d:
            path = Path(d) / "invented.json"
            raw = b'{"invented": true}\n'
            path.write_bytes(raw)
            ref = {
                "path": str(path.relative_to(m.ROOT)),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
            }
            self.assertEqual(m.verify_record(ref), raw)
            for change in (
                {"sha256": "0" * 64},
                {"size_bytes": len(raw) + 1},
                {"path": "../outside.json"},
            ):
                with self.subTest(change=change), self.assertRaises(ValueError):
                    m.verify_record({**ref, **change})


if __name__ == "__main__":
    unittest.main()
