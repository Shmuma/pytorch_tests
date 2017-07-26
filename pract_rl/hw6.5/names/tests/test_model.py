import unittest

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
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

    def test_train(self):
        """
        Perform train on very simple dataset, consisting only with single example
        """
        data = ['aa']
        encoder = input.InputEncoder(data)
        net = model.Model(len(encoder), hidden_size=10)
        objective = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1)
        max_loss = min_loss = None

        for _ in range(100):
            net.zero_grad()
            packed_seq, true_indices = input.batch_to_train(data, encoder)
            out, _ = net(packed_seq)
            v_loss = objective(out, true_indices)
            v_loss.backward()
            optimizer.step()
            l = v_loss.data[0]
            if max_loss is None:
                max_loss = min_loss = l
            else:
                max_loss = max(max_loss, l)
                min_loss = min(min_loss, l)

        loss_drop = max_loss / min_loss
        self.assertGreater(loss_drop, 10.0)

        # try to predict letters
        out = model.inference_one(encoder.START_TOKEN, encoder, net)
        self.assertEquals(out, 'a')
        out = model.inference_one('a', encoder, net)
        self.assertEquals(out, 'a')
        out = model.inference_one('aa', encoder, net)
        self.assertEquals(out, encoder.END_TOKEN)

    def test_generate(self):
        data = ['aa']
        encoder = input.InputEncoder(data)
        net = model.Model(len(encoder), hidden_size=10)
        objective = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.1)
        max_loss = min_loss = None

        for _ in range(100):
            net.zero_grad()
            packed_seq, true_indices = input.batch_to_train(data, encoder)
            out, _ = net(packed_seq)
            v_loss = objective(out, true_indices)
            v_loss.backward()
            optimizer.step()
            l = v_loss.data[0]
            if max_loss is None:
                max_loss = min_loss = l
            else:
                max_loss = max(max_loss, l)
                min_loss = min(min_loss, l)

        res = model.generate_name(net, encoder)
        self.assertEqual('aa', res)

if __name__ == '__main__':
    unittest.main()
