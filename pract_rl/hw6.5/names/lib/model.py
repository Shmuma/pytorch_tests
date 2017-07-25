import logging

import torch.nn as nn


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

        out = self.out(rnn_out.data)
        return out

