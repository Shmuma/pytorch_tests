import unittest
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

from char_rnn import data
from char_rnn import model


class TestModel(unittest.TestCase):
    def test_name_to_tensor(self):
        res = model.name_to_tensor("a")
        self.assertTrue(torch.is_tensor(res))
        self.assertEqual((1, 1, len(data.ALL_ASCII_LETTERS)), res.size())
        s = res.numpy().sum()
        self.assertEqual(1, s)

    def test_pytorch_pack_padded(self):
        t = Variable(torch.FloatTensor([[[1, 2, 3, 4]], [[5, 6, 7, 0]]]))
        s = t.size()
        r = rnn_utils.pack_padded_sequence(t, [4, 3], batch_first=True)
        print(r)
        rnn = nn.RNN(input_size=4, hidden_size=10, num_layers=1)
        out, hid = rnn.forward(r, None)
        print(r)

pass
