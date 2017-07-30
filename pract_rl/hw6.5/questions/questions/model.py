import math as m
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

from . import data


class FixedEmbeddingsModel(nn.Module):
    def __init__(self, embeddings, hidden_size, h_softmax=False):
        super(FixedEmbeddingsModel, self).__init__()

        dict_size, emb_size = embeddings.shape
        # create fixed embeddings layer
        self.embeddings = nn.Embedding(dict_size, emb_size)
        self.embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        self.embeddings.weight.requires_grad = False

        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                           batch_first=True, dropout=0.5)
        if h_softmax:
            out_size = 2**m.ceil(m.log(dict_size, 2)) - 1
        else:
            out_size = dict_size
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, x, h=None):
        if isinstance(x, rnn_utils.PackedSequence):
            x_emb = self.embeddings(x.data)
            rnn_in = rnn_utils.PackedSequence(data=x_emb, batch_sizes=x.batch_sizes)
        else:
            rnn_in = self.embeddings(x)
        rnn_out, rnn_hidden = self.rnn(rnn_in, h)
        if isinstance(rnn_out, rnn_utils.PackedSequence):
            out = self.out(rnn_out.data)
        else:
            out = self.out(rnn_out[0])
        return out, rnn_hidden


class HierarchicalSoftmaxLoss(nn.Module):
    """
    On input we get n-1 raw scores for binary tree (n == vocabulary size) and list of valid class indices
    Scores are interpreted for LEFT node
    1. apply sigmoid layer
    2. 
    """
    def __init__(self):
        super(HierarchicalSoftmaxLoss, self).__init__()

    def forward(self, scores, class_indices, cuda=False):
        batch_size, dict_size = scores.size()
        tree_probs = torch.sigmoid(scores)
        pad = Variable(torch.ones(batch_size, 1))
        if cuda:
            pad = pad.cuda()
        padded_probs = torch.cat([pad, tree_probs], dim=1)

        code_len = m.ceil(m.log(dict_size, 2))
        mask = 2**(code_len-1)
        level = 1
        left_indices = []
        right_indices = []
        cur_index = 1

        while mask > 0:
            left_list = []
            right_list = []
            for right_branch in map(lambda v: bool(v & mask), class_indices):
                if not right_branch:
                    left_list.append(cur_index)
                    right_list.append(0)
                else:
                    left_list.append(0)
                    right_list.append(cur_index+1)
            left_indices.append(left_list)
            right_indices.append(right_list)
            cur_index += level
            level <<= 1
            mask >>= 1

        left_t = Variable(torch.LongTensor(left_indices)).t()
        right_t = Variable(torch.LongTensor(right_indices)).t()
        if cuda:
            left_t = left_t.cuda()
            right_t = right_t.cuda()

        left_part = torch.gather(padded_probs, dim=1, index=left_t)
        left_p = torch.prod(left_part, dim=1)
        right_part = torch.gather(padded_probs, dim=1, index=right_t)
        right_p = torch.prod(right_part, dim=1)
        probs = torch.mul(left_p, right_p)
        return torch.sum(-probs.log()) / batch_size


def generate_question(net, word_dict, rev_word_dict, cuda=False):
    """
    Sample question from model
    :param net: model to used
    :param word_dict: word -> idx mapping 
    :param rev_word_dict: idx -> word mapping 
    :return: list of question tokens
    """
    assert isinstance(net, nn.Module)
    assert isinstance(word_dict, dict)
    assert isinstance(rev_word_dict, dict)

    result = []
    hidden = None
    token = data.END_TOKEN
    sm = nn.Softmax()

    while len(result) < 200:
        token_v = Variable(torch.LongTensor([[word_dict[token]]]))
        if cuda:
            token_v = token_v.cuda()
        out, hidden = net(token_v, hidden)
        out = sm(out)
        out = out.data.cpu().numpy()
        idx = np.random.choice(len(word_dict), p=out[0])
        token = rev_word_dict[idx]
        if token == data.END_TOKEN:
            break
        result.append(token)

    return result
