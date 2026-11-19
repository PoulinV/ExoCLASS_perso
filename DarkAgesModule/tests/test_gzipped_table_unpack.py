"""The gzipped spectral-distortion table is unpacked once, byte-identically."""

import gzip
import os
import sys
import tempfile
import unittest
from pathlib import Path

DARKAGES_BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault("DARKAGES_BASE", str(DARKAGES_BASE))
sys.path.insert(0, str(DARKAGES_BASE))

from DarkAges import _unpack_gzipped_table  # noqa: E402


class GzippedTableUnpackTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.table = os.path.join(self.tmp.name, "table.dat")
        self.payload = b"".join(b"%d 1.5e-3 2.0 3.0 4.0\n" % i for i in range(5000))

    def tearDown(self):
        self.tmp.cleanup()

    def _write_packed(self, payload):
        with gzip.open(self.table + ".gz", "wb") as fh:
            fh.write(payload)

    def test_unpacks_when_only_the_gz_exists(self):
        self._write_packed(self.payload)
        _unpack_gzipped_table(self.table)
        self.assertEqual(Path(self.table).read_bytes(), self.payload)
        self.assertEqual(sorted(os.listdir(self.tmp.name)), ["table.dat", "table.dat.gz"])

    def test_existing_table_is_left_untouched(self):
        Path(self.table).write_bytes(b"local")
        self._write_packed(self.payload)
        _unpack_gzipped_table(self.table)
        self.assertEqual(Path(self.table).read_bytes(), b"local")

    def test_nothing_happens_without_either_file(self):
        _unpack_gzipped_table(self.table)
        self.assertEqual(os.listdir(self.tmp.name), [])

    def test_corrupt_archive_leaves_no_partial_file(self):
        Path(self.table + ".gz").write_bytes(b"not a gzip stream")
        with self.assertRaises(OSError):
            _unpack_gzipped_table(self.table)
        self.assertEqual(os.listdir(self.tmp.name), ["table.dat.gz"])


if __name__ == "__main__":
    unittest.main()
