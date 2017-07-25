import unittest

from lib import input


class TestEncoder(unittest.TestCase):
    def test_simple(self):
        enc = input.InputEncoder(['abc'])
        self.assertEqual(3, len(enc))

        d = enc.encode("aaa")
        self.assertEqual(d.shape, (3, len(enc)))
        self.assertEqual(d[0][0], 1.0)

        d = enc.encode("a", width=4)
        self.assertEqual(d.shape, (4, len(enc)))


if __name__ == '__main__':
    unittest.main()
