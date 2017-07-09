import os
import tempfile
import unittest

from char_rnn import data


class TestData(unittest.TestCase):
    def test_all_filenames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            p1 = os.path.join(temp_dir, "file1.txt")
            p2 = os.path.join(temp_dir, "file2.txt")
            p3 = os.path.join(temp_dir, "file.dat")
            for p in [p1, p2, p3]:
                with open(p, "w+") as fd:
                    fd.write('1')
            names = data.all_filenames(temp_dir)
            self.assertIsInstance(names, list)
            self.assertEqual(len(names), 2)
            self.assertEqual(sorted(names), [p1, p2])

