import logging

import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


class Model(nn.Module):
    log = logging.getLogger("Model")

    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden=None):
        """
        NB: softmax is not needed, as we're using CrossEntropyLoss which already does softmax
        :param x:
        :param hidden:
        :return:
        """
        rnn_out, rnn_hidden = self.rnn(x, hidden)

        if isinstance(rnn_out, rnn_utils.PackedSequence):
            rnn_out = rnn_out.data
        out = self.out(rnn_out)
        return out


def inference_one(s, encoder, net):
    """
    Predict next character in a sequence
    :param s: input string
    :param encoder: Encoder to use
    :param net: Model
    :return: next character predicted by the model
    """
    v = encoder.encode(s)
    v_v = Variable(torch.from_numpy(np.array([v])))
    v_s = rnn_utils.pack_padded_sequence(v_v, [len(s)+2], batch_first=True)
    res = net(v_s)
    softmax = nn.Softmax()
    res = softmax(res)
    out_v = res[len(s)]
    _, out_idx = torch.max(out_v.data, dim=0)
    return encoder.indices_to_chars(out_idx)[0]
