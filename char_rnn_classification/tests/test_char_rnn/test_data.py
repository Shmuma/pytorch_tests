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

    def test_name_to_ascii(self):
        self.assertEqual("test", data.name_to_ascii("test"))
        self.assertEqual("Slusarski", data.name_to_ascii("Ślusàrski"))
        self.assertEqual("Test, and another test", data.name_to_ascii("Test, and another test"))

    def test_read_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            p = os.path.join(temp_dir, "file1.txt")
            with open(p, "w+") as fd:
                fd.write('test\n')
                fd.write('another\n')
            res = data.read_file(p)
            self.assertIsInstance(res, list)
            self.assertEqual(2, len(res))
            self.assertEqual(res, ['test', 'another'])

    def test_read_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            p1 = os.path.join(temp_dir, "file1.txt")
            p2 = os.path.join(temp_dir, "file2.txt")
            for p in [p1, p2]:
                with open(p, "w+") as fd:
                    fd.write('name\n')
                    fd.write('aname\n')
            res = data.read_files([p1, p2])
            self.assertIsInstance(res, dict)
            self.assertTrue('file1' in res)
            self.assertEqual(['name', 'aname'], res['file1'])
