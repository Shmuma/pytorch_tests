#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


if __name__ == "__main__":
    # batch = 2
    # max_seq_len = 4
    t = Variable(torch.FloatTensor([[[1], [2], [3], [4]], [[5], [6], [7], [0]]]))
    print(t.size())
    r = rnn_utils.pack_padded_sequence(t, [4, 3], batch_first=True)
    print(r)
    rnn = nn.RNN(input_size=1, hidden_size=10, num_layers=1, batch_first=True)
    out, hid = rnn.forward(r, None)

#    print("out", out)
#    print("hid", hid)
    unpacked, unpacked_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
    print("Unpacked output", unpacked.size())
    print("Unpacked output lengths", unpacked_len)
    print(hid)
