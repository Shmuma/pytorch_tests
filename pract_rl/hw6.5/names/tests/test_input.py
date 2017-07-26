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

    def test_end_token(self):
        enc = input.InputEncoder(['abc'])
        r = enc.end_token()
        self.assertEqual(5, r.shape[0])

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

    def test_batch_to_train(self):
        enc = input.InputEncoder(['abc'])
        packed_seq, true_vals = input.batch_to_train(['a', 'b', 'c'], enc)
        self.assertEqual(true_vals.data.numpy(), [   1, 4,
                                                  0, 2, 4,
                                                  0, 3, 4, 4])



if __name__ == '__main__':
    unittest.main()
