import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=False, dropout=0.5,
                           num_layers=1)

    def forward(self, x, h=None):
        """
        Return output and hidden state suitable to be fed into decoder's input, i.e. of size
        batch, dim

        Output could be used for attention
        """
        out, hidden = self.rnn(x, h)
        if True:
            # for LSTM, we have hidden state and cell state tuple,
            # which we need to concat
            hidden = torch.cat(hidden, dim=0)
        # swap dimensions to make batch first dim
        hidden = torch.transpose(hidden, 0, 1)
        # flatten on last two dimensions
        hidden = hidden.clone()
        hidden = hidden.view(-1, hidden.size()[1]*hidden.size()[2])
        print(out)
        return hidden

    def state_size(self):
        # two layers, two direction * 2 (lstm hidden and cell)
        return self.hidden_size * 2


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.5, num_layers=1)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        if isinstance(x, rnn_utils.PackedSequence):
            y, h = self.rnn(x, h)
            ys = y.data
        else:
            # input has to be batch*embedding, i.e. contain only one time stamp
            y, h = self.rnn(x.unsqueeze(dim=1), h)
            ys = y.squeeze(dim=1)
        out = self.out(ys)
        return out, h

    def reorder_hidden(self, hidden, order):
        """
        Reorder hidden along the batch dimension
        :param hidden:
        :param order:
        :return:
        """
        if isinstance(hidden, tuple):
            return tuple(torch.index_select(v, dim=1, index=order) for v in hidden)

pass
