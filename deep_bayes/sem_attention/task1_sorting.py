#!/usr/bin/env python3
"""
Implementation of sorting network with attention.

Original tutorial is on lasagna:
https://github.com/bayesgroup/deepbayes2017/blob/master/sem3-attention/Attention_seminar%20(Start%20here).ipynb
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


MAX_TIME_LENGTH = 10
CODE_SIZE = 10
DECODER_HIDDEN = 32
ATTENTION_HIDDEN = 16


def generate_sample(min_length=3, max_length=10, code_size=CODE_SIZE):
    assert code_size >= max_length
    length = np.random.randint(min_length, max_length)

    keys = np.random.permutation(length)
    values = np.random.permutation(length)
    input_pairs = list(zip(keys, values))

    input_1hot = np.zeros([length + 1, code_size * 2], dtype=np.float32)
    for i, (k, v) in enumerate(input_pairs):
        input_1hot[i + 1][k] = 1
        input_1hot[i + 1][code_size + v] = 1

    sorted_pairs = sorted(input_pairs, key=lambda p: p[0])

    target_1hot = np.zeros([length + 1, code_size * 2], dtype=np.float32)
    for i, (k, v) in enumerate(sorted_pairs):
        target_1hot[i + 1][k] = 1
        target_1hot[i + 1][code_size + v] = 1

    return input_1hot, target_1hot


def test_sample():
    inp, out = generate_sample(max_length=5, code_size=5)
    print('-' * 9 + "KEY" + '-' * 9 + ' ' + '+' * 9 + "VAL" + "+" * 9)
    print("Input pairs:\n", inp)
    print("Target pairs:\n", out)


class AttentionWeights(nn.Module):
    def __init__(self, size_enc_hidden, size_dec_hidden, size_att_hidden):
        """
        Create attention weights layer. It's responsibility is to convert encoder sequence
        into vector of attention weights
        :param size_enc_hidden: size of hidden encoder state
        :param size_dec_hidden: size of hidden decoder state (aka query)
        :param size_att_hidden: size of attention hidden state (middle dimension)
        """
        super(AttentionWeights, self).__init__()
        self.w_enc = nn.Linear(size_enc_hidden, size_att_hidden, bias=False)
        self.w_query = nn.Linear(size_dec_hidden, size_att_hidden, bias=False)
        self.w_out = nn.Linear(size_att_hidden, 1, bias=False)

    def forward(self, encoder_sequence, query):
        """
        Perform weights calculations
        :param encoder_sequence: tensor of size batch*time*enc_hidden with encoder outputs we're going to attend
        :param query: tensor of size batch*dec_hidden of attention query
        :return: tensor of size batch*time with attention weights
        """
        query_part = self.w_query(query)
        enc_part = self.w_enc(encoder_sequence)
        # here we do broadcast of query by time (dim=1)
        part = enc_part + query_part
        part = F.tanh(part)
        logits = self.w_out(part).squeeze(dim=2)
        return F.softmax(logits)


def initial_query(batch_size, hidden_size):
    t = np.zeros((batch_size, hidden_size), dtype=np.float32)
    return Variable(torch.from_numpy(t))


if __name__ == "__main__":
    test_sample()
    batch_input, batch_target = generate_sample()
    # batch of size 1
    batch_input = np.expand_dims(batch_input, axis=0)
    batch_input_v = Variable(torch.from_numpy(batch_input))
    print(batch_input.shape)
#    enc_res, _ = encoder(batch_input_v)
#    print(enc_res.size())
    # input is original encoder seq plus result of attention, each of them is CODE_SIZE*2
    rnn = nn.RNN(input_size=CODE_SIZE*2*2, hidden_size=DECODER_HIDDEN)
    attention_weights = AttentionWeights(size_enc_hidden=CODE_SIZE*2, size_dec_hidden=DECODER_HIDDEN,
                                         size_att_hidden=ATTENTION_HIDDEN)

    query_v = initial_query(batch_size=1, hidden_size=DECODER_HIDDEN)

    att_scores = attention_weights(batch_input_v, query_v)
    print(att_scores)
    pass
