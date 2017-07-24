#!/usr/bin/env python3
import numpy as np
import random
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


TRAIN_FILE = "names"
HIDDEN_SIZE = 128

log = logging.getLogger("train")


class Model(nn.Module):
    log = logging.getLogger("Model")

    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, input_size)
        self.softmax = nn.Softmax()

    def forward(self, x, hidden=None):
        rnn_out, rnn_hidden = self.rnn(x, hidden)

        out = self.out(rnn_out.data)
        out = self.softmax(out)
        return out


class InputEncoder:
    def __init__(self, data):
        self.alphabet = list(sorted(set("".join(data))))
        self.idx = {c: pos for pos, c in enumerate(self.alphabet)}

    def __len__(self):
        return len(self.alphabet)

    def encode(self, s, width=None):
        assert isinstance(s, str)

        if width == None:
            width = len(s)
        res = np.zeros((width, len(self)), dtype=np.float32)
        for sample_idx, c in enumerate(s):
            idx = self.idx[c]
            res[sample_idx][idx] = 1.0

        return res



def read_data(file_name=TRAIN_FILE, prepend_space=True):
    with open(file_name, "rt", encoding='utf-8') as fd:
        res = map(str.strip, fd.readlines())
        if prepend_space:
            res = map(lambda s: ' ' + s, res)
        return list(res)



def test_model():
    encoder = InputEncoder(['abcde'])
    assert len(encoder) == 5
    items = ["abc", "ab"]
    lens = list(map(len, items))

    data = np.array([
        encoder.encode(s, width=max(lens))
        for s in items
    ])

    v_data = Variable(torch.from_numpy(data))
    seq = rnn_utils.pack_padded_sequence(v_data, lens, batch_first=True)

    log.info(data)
    model = Model(len(encoder), HIDDEN_SIZE)
    log.info(model)
    res = model.forward(seq)
    log.info(res)



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)-14s %(message)s", level=logging.INFO)

    test_model()

    data = read_data()
    input_encoder = InputEncoder(data)

    log.info("Read %d train samples, first 10: %s", len(data), ', '.join(data[:10]))
    pass
