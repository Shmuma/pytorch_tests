import torch
import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)

    def forward(self, x, h=None):
        return self.rnn(x, h)


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.rnn = nn.RNN(output_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.sm = nn.Softmax()

    def forward(self, x, h):
        y, h = self.rnn(x, h)
        out = self.out(y)
        out = self.sm(out)
        return out, h

pass
