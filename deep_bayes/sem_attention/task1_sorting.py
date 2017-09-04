#!/usr/bin/env python3
"""
Implementation of sorting network with attention.

Original tutorial is on lasagna:
https://github.com/bayesgroup/deepbayes2017/blob/master/sem3-attention/Attention_seminar%20(Start%20here).ipynb
"""
from tqdm import tnrange
import itertools
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


MAX_TIME_LENGTH = 10
CODE_SIZE = 10
DECODER_HIDDEN = 32
ATTENTION_HIDDEN = 64


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


def attention_outputs(encoder_sequence, att_weights):
    """
    Calculate output of attention with given weights
    :param encoder_sequence: tensor of size batch*time*enc_hidden with sequence we're going to attend
    :param att_weights: tensor with attention weights. Size is batch*time
    :return: tensor of batch*enc_hidden with result of attention
    """
    return torch.bmm(encoder_sequence.transpose(1, 2), att_weights.unsqueeze(dim=2)).squeeze(dim=2)


def initial_query(batch_size, hidden_size):
    t = np.zeros((batch_size, hidden_size), dtype=np.float32)
    return Variable(torch.from_numpy(t))


if __name__ == "__main__":
    # input is original encoder seq plus result of attention, each of them is CODE_SIZE*2
    rnn = nn.RNN(input_size=CODE_SIZE*2*2, hidden_size=DECODER_HIDDEN, batch_first=True)
    output_layer = nn.Linear(in_features=DECODER_HIDDEN, out_features=CODE_SIZE*2)
    attention_weights = AttentionWeights(size_enc_hidden=CODE_SIZE*2, size_dec_hidden=DECODER_HIDDEN,
                                         size_att_hidden=ATTENTION_HIDDEN)
    optimizer = optim.Adam(params=itertools.chain(rnn.parameters(), output_layer.parameters(),
                                                  attention_weights.parameters()),
                           lr=0.01)
    objective = nn.BCELoss()
    min_loss = None

    for i in tnrange(10000):
        optimizer.zero_grad()
        batch_input, batch_target = generate_sample()
        # batch of size 1
        batch_input = np.expand_dims(batch_input, axis=0)
        batch_input_v = Variable(torch.from_numpy(batch_input))
        batch_target = np.expand_dims(batch_target, axis=0)
        batch_target_v = Variable(torch.from_numpy(batch_target))

        # initial query for first item
        query_v = initial_query(batch_size=1, hidden_size=DECODER_HIDDEN)

        total_loss = Variable(torch.FloatTensor([0.0]))

        # iterate over our batch sequence
        for ofs in range(batch_input_v.size()[1]-1):
            att_weights = attention_weights(batch_input_v, query_v)
            att_outputs = attention_outputs(batch_input_v, att_weights)
            encoder_input = batch_input_v[:, ofs]
            desired_output = batch_target_v[:, ofs+1]
            # build rnn input and create time dimension
            rnn_in = torch.cat([encoder_input, att_outputs], dim=1).unsqueeze(dim=1)
            rnn_out, rnn_hid = rnn(rnn_in, query_v.unsqueeze(dim=0))
            # first dim in rnn hidden is num_layers*num_directions, so, squeeze it
            query_v = rnn_hid.squeeze(dim=0)
            # squeeze time of output
            rnn_out = rnn_out.squeeze(dim=1)
            output = F.sigmoid(output_layer(rnn_out))
            loss = objective(output, desired_output)
            total_loss += loss
        total_loss /= batch_input_v.size()[1]
        total_loss.backward()
        optimizer.step()
        total_loss_val = total_loss.cpu().data.numpy()[0]
        min_loss = min(min_loss, total_loss_val) if min_loss is not None else total_loss_val

        if i % 500 == 0:
            print("loss: %.6f\tmin loss: %.6f" % (total_loss_val, min_loss))

    pass
