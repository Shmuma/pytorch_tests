import math as m
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

from . import data


class FixedEmbeddingsModel(nn.Module):
    """
    Model maps input indices to output of hidden size
    """
    def __init__(self, embeddings, hidden_size):
        super(FixedEmbeddingsModel, self).__init__()

        dict_size, emb_size = embeddings.shape
        # create fixed embeddings layer
        self.embeddings = nn.Embedding(dict_size, emb_size)
        self.embeddings.weight.data.copy_(torch.from_numpy(embeddings))
        self.embeddings.weight.requires_grad = False

        self.rnn = nn.LSTM(input_size=emb_size, hidden_size=hidden_size,
                           batch_first=True, dropout=0.5)

    def forward(self, x, h=None):
        if isinstance(x, rnn_utils.PackedSequence):
            x_emb = self.embeddings(x.data)
            rnn_in = rnn_utils.PackedSequence(data=x_emb, batch_sizes=x.batch_sizes)
        else:
            rnn_in = self.embeddings(x)
        return self.rnn(rnn_in, h)


class MappingModule(nn.Module):
    """
    Base parent for language model output with methods:
    1. forward() -> returns loss for batch
    2. infer() -> returns indices for inferred words
    """
    def forward(self, x, valid_indices):
        raise NotImplemented

    def infer(self, x):
        raise NotImplemented

#
class SoftmaxMappingModule(MappingModule):
    """
    Classical softmax
    """
    def __init__(self, input_size, embeddings_size):
        super(SoftmaxMappingModule, self).__init__()
        self.out = nn.Linear(input_size, embeddings_size)
        self.loss = nn.CrossEntropyLoss()
        self.sm = nn.Softmax()

    def forward(self, x, valid_indices):
        """
        Calculate cross entropy loss
        :param x: input vector
        :param valid_indices: valid indices
        :return: loss
        """
        predicted = self.out(x)
        return self.loss(predicted, valid_indices)

    def infer(self, x):
        """
        From net's output return indices of predicted words
        :param x: batch of outputs
        :return:
        """
        out = self.out(x)
        out = self.sm(out)
        res = torch.multinomial(out, 1)
        return res


class HierarchicalSoftmaxMappingModule(nn.Module):
    """
    Build full tree for hierarchical softmax. Not the best option from performance point of view,
    but wasted too much time making it working, so, let's keep it here :)
    """
    def __init__(self):
        super(HierarchicalSoftmaxMappingModule, self).__init__()

    def forward(self, scores, class_indices, cuda=False):
        batch_size, dict_size = scores.size()
        # without initial sigmoid, our predefined values should be large and small vals
        pad_one = Variable(torch.ones(batch_size, 1) * 100.0)
        pad_zer = Variable(torch.ones(batch_size, 1) * (-100.0))
        if cuda:
            pad_one = pad_one.cuda()
            pad_zer = pad_zer.cuda()
        padded_scores = torch.cat([pad_one, pad_zer, scores], dim=1)

        code_len = m.ceil(m.log(dict_size, 2))
        mask = 2**(code_len-1)
        level = 1
        left_indices = []
        right_indices = []
        indices = [2] * batch_size

        while mask > 0:
            left_list = []
            right_list = []
            for batch_idx, right_branch in enumerate(map(lambda v: bool(v & mask), class_indices)):
                cur_index = indices[batch_idx]
                if not right_branch:
                    left_list.append(cur_index)
                    right_list.append(1)
                    cur_index <<= 1
                    cur_index -= 1
                else:
                    left_list.append(0)
                    right_list.append(cur_index)
                    cur_index <<= 1
                indices[batch_idx] = cur_index
            left_indices.append(left_list)
            right_indices.append(right_list)
            level <<= 1
            mask >>= 1

        left_t = Variable(torch.LongTensor(left_indices)).t()
        right_t = Variable(torch.LongTensor(right_indices)).t()
        if cuda:
            left_t = left_t.cuda()
            right_t = right_t.cuda()

        left_part = torch.sigmoid(torch.gather(padded_scores, dim=1, index=left_t))
        right_part = 1.0 - torch.sigmoid(torch.gather(padded_scores, dim=1, index=right_t))
        left_p = torch.prod(left_part, dim=1)
        right_p = torch.prod(right_part, dim=1)
        probs = torch.mul(left_p, right_p)
        return torch.sum(-probs.log()) / batch_size


class TwoLevelSoftmaxMappingModule(MappingModule):
    def __init__(self, input_size, dict_size, freq_ratio=0.2, count_of_classes=5):
        """
        Construct two level softmax loss. Dictionary should be ordered by decrease of frequency
        :param dict_size: count of words in the vocabulary
        :param hidden_size: size of input from hidden layer
        :param freq_ratio: ratio of frequent words to keep in top-level classifier
        :param count_of_classes: how many groups to create in second level
        """
        super(TwoLevelSoftmaxMappingModule, self).__init__()
        self.dict_size = dict_size
        self.input_size = input_size
        self.count_freq = int(dict_size * freq_ratio)
        self.count_of_classes = count_of_classes

        # build layers
        self.level_one = nn.Linear(input_size, self.count_freq + self.count_of_classes)
        self.level_two = []
        words_left = dict_size - self.count_freq
        chunk = words_left // count_of_classes
        for idx in range(count_of_classes):
            words_left -= chunk
            this_chunk = chunk
            if words_left < chunk:
                this_chunk += words_left
            level = nn.Linear(input_size, this_chunk)
            self.level_two.append(level)
            # to make or level layer visible to .parameters()
            setattr(self, "level_two_%d" % idx, level)

        self.sm = nn.Softmax()

    def forward(self, x, valid_indices):
        out_one = self.sm(self.level_one(x))
        next_layer_mask = valid_indices >= self.count_freq
        one_indices = valid_indices.clone()
        one_indices.masked_fill_(next_layer_mask, 0)
        one_probs = torch.masked_select(out_one, one_indices)
#        one_probs = out_one[one_indices.data]

        pass




def generate_question(net, net_map, word_dict, rev_word_dict, cuda=False):
    """
    Sample question from model
    :param net: model to used
    :param net_map: mapping module to get word index
    :param word_dict: word -> idx mapping
    :param rev_word_dict: idx -> word mapping
    :return: list of question tokens
    """
    assert isinstance(net, nn.Module)
    assert isinstance(net_map, MappingModule)
    assert isinstance(word_dict, dict)
    assert isinstance(rev_word_dict, dict)

    result = []
    hidden = None
    token = data.END_TOKEN

    while len(result) < 200:
        token_v = Variable(torch.LongTensor([[word_dict[token]]]))
        if cuda:
            token_v = token_v.cuda()
        out, hidden = net(token_v, hidden)
        idx = net_map.infer(out[0])
        idx = idx.data.cpu().numpy()[0][0]
        token = rev_word_dict[idx]
        if token == data.END_TOKEN:
            break
        result.append(token)

    return result
