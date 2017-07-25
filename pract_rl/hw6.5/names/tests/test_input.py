import unittest

from lib import input


class TestData(unittest.TestCase):
    def test_simple(self):
        enc = input.InputEncoder(['abc'])
        self.assertEqual(5, len(enc))       # 3 + start + end

        d = enc.encode("aaa")
        self.assertEqual(d.shape, (5, len(enc)))
        self.assertEqual(d[0][0], 1.0)
        self.assertEqual(d[1][2], 1.0)

        d = enc.encode("a", width=4)
        self.assertEqual(d.shape, (6, len(enc)))

    def test_iterate_batches(self):
        data = ["abc", "cde", "e", "f"]
        r = list(input.iterate_batches(data, batch_size=2, shuffle=False))
        self.assertEqual(2, len(r))
        self.assertEqual(["abc", "cde"], r[0])
        self.assertEqual(["e", "f"], r[1])

        r = list(input.iterate_batches(data, batch_size=3, shuffle=False))
        self.assertEqual(2, len(r))
        self.assertEqual(3, len(r[0]))
        self.assertEqual(1, len(r[1]))


if __name__ == '__main__':
    unittest.main()
