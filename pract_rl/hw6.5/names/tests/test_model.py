import unittest

import numpy as np

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

from lib import model, input


class ModelTest(unittest.TestCase):
    def test_simple(self):
        encoder = input.InputEncoder(['abcde'])
        items = ["abc", "ab"]
        lens = list(map(len, items))

        dat = np.array([
            encoder.encode(s, width=max(lens))
            for s in items
        ])

        v_dat = Variable(torch.from_numpy(dat))
        seq = rnn_utils.pack_padded_sequence(v_dat, lens, batch_first=True)

        m = model.Model(len(encoder), 16)
        res = m.forward(seq)


if __name__ == '__main__':
    unittest.main()
