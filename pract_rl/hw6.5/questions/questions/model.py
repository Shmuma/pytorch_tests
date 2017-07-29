import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

from . import data


class FixedEmbeddingsModel(nn.Module):
    def __init__(self, embeddings, hidden_size):
        super(FixedEmbeddingsModel, self).__init__()

        # create fixed embeddings layer
        self.embeddings = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embeddings.weight.data.copy_(torch.from_numpy(embeddings))
#        self.embeddings.weight.requires_grad = False

        self.rnn = nn.LSTM(input_size=embeddings.shape[1], hidden_size=hidden_size,
                           batch_first=True, dropout=0.5, bidirectional=False)
        self.out = nn.Linear(hidden_size, embeddings.shape[0])

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
    token = data.START_TOKEN
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
