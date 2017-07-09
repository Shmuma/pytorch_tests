import unittest
import torch

from char_rnn import data
from char_rnn import model


class TestModel(unittest.TestCase):
    def test_name_to_tensor(self):
        res = model.name_to_tensor("a")
        self.assertTrue(torch.is_tensor(res))
        self.assertEqual((1, 1, len(data.ALL_ASCII_LETTERS)), res.size())
        s = res.numpy().sum()
        self.assertEqual(1, s)

pass
