import unittest

from lib import input


class TestEncoder(unittest.TestCase):
    def test_simple(self):
        enc = input.InputEncoder(['abc'])
        self.assertEqual(5, len(enc))       # 3 + start + end

        d = enc.encode("aaa")
        self.assertEqual(d.shape, (5, len(enc)))
        self.assertEqual(d[0][0], 1.0)
        self.assertEqual(d[1][2], 1.0)

        d = enc.encode("a", width=4)
        self.assertEqual(d.shape, (6, len(enc)))


if __name__ == '__main__':
    unittest.main()
