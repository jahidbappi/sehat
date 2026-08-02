"""Tests for sehat.data.download — checksums, resume safety, idempotency.

Network-free: downloads are served from ``file://`` URLs pointing at temp
files. Runnable with either ``pytest`` or ``python -m unittest``.
"""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sehat.data.download import ChecksumMismatchError, download_file, sha256_file, verify_checksum

CONTENT = b"sehat test payload " * 1000
CONTENT_SHA256 = hashlib.sha256(CONTENT).hexdigest()


class Sha256Test(unittest.TestCase):
    def test_digest_matches_hashlib(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.bin"
            path.write_bytes(CONTENT)
            self.assertEqual(sha256_file(path), CONTENT_SHA256)
            self.assertTrue(verify_checksum(path, CONTENT_SHA256.upper()))
            self.assertFalse(verify_checksum(path, "0" * 64))


class DownloadFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.source = self.tmp / "source.bin"
        self.source.write_bytes(CONTENT)
        self.url = self.source.as_uri()

    def test_downloads_and_verifies(self) -> None:
        dest = download_file(self.url, self.tmp / "out" / "f.bin", sha256=CONTENT_SHA256)
        self.assertEqual(dest.read_bytes(), CONTENT)
        self.assertFalse((self.tmp / "out" / "f.bin.part").exists())

    def test_checksum_mismatch_raises_and_keeps_part(self) -> None:
        dest = self.tmp / "f.bin"
        with self.assertRaises(ChecksumMismatchError):
            download_file(self.url, dest, sha256="0" * 64)
        self.assertFalse(dest.exists())
        self.assertEqual((self.tmp / "f.bin.part").read_bytes(), CONTENT)

    def test_existing_verified_file_is_not_redownloaded(self) -> None:
        dest = download_file(self.url, self.tmp / "f.bin", sha256=CONTENT_SHA256)
        self.source.write_bytes(b"corrupted source")
        again = download_file(self.url, dest, sha256=CONTENT_SHA256)
        self.assertEqual(again.read_bytes(), CONTENT)

    def test_leftover_part_does_not_poison_fresh_download(self) -> None:
        dest = self.tmp / "f.bin"
        (self.tmp / "f.bin.part").write_bytes(b"partial garbage")
        result = download_file(self.url, dest, sha256=CONTENT_SHA256)
        self.assertEqual(result.read_bytes(), CONTENT)


if __name__ == "__main__":
    unittest.main()
