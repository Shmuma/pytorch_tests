import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


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
        return out, hidden

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


class AttentionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_encoder_input):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.max_encoder_input = max_encoder_input
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=0.5, num_layers=1)
        self.out_char = nn.Linear(hidden_size, output_size)
        self.out_attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h, packed_encoder_output):
        if isinstance(x, rnn_utils.PackedSequence):
            y, h = self.rnn(x, h)
            ys = y.data
        else:
            y, h = self.rnn(x.unsqueeze(dim=1), h)
            ys = y.squeeze(dim=1)
        rnn_hid, rnn_cell = h
        rnn_hid = rnn_hid.squeeze(dim=0)
        out_char = self.out_char(ys)
        out_attn = self._calc_attention(rnn_hid, packed_encoder_output)
        return out_char, out_attn, h

    def _calc_attention(self, rnn_hidden, packed_encoder_output):
        """
        Calculate attention output based on decoder and encoder hidden states
        :param rnn_hidden: tensor of batch * hidden_size with hidden state of decoder
        :param packed_encoder_output: PackedSequence with output from encoder. Max len could be less than max_encoder_input
        :return: tensor of batch * hidden_size
        """
        attn_vals = self.out_attn(rnn_hidden)
        padded_output, encoder_output_lens = rnn_utils.pad_packed_sequence(packed_encoder_output, batch_first=True)
        scores = torch.bmm(padded_output, attn_vals.unsqueeze(dim=2))
        scores = scores.squeeze(dim=2)
        scores = F.softmax(scores)
        scores = scores.unsqueeze(dim=2)
        scores = scores.expand_as(padded_output)
        scaled_output = torch.mul(padded_output, scores)
        output = scaled_output.mean(dim=1).squeeze(dim=1)
        return output

    def initial_attention(self, batch_size, cuda=False):
        res = Variable(torch.zeros(batch_size, self.hidden_size))
        if cuda:
            res = res.cuda()
        return res

pass
