import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

from . import data

INPUT_SIZE = len(data.ALL_ASCII_LETTERS)
LETTERS_INDICES = {c: idx for idx, c in enumerate(data.ALL_ASCII_LETTERS)}


def char_to_tensor(chr):
    res = torch.zeros((1, INPUT_SIZE))
    pos = data.ALL_ASCII_LETTERS.find(chr)
    res[0][pos] = 1
    return res


def name_to_tensor(name):
    res = torch.zeros((len(name), 1, INPUT_SIZE))
    for idx, c in enumerate(name):
        pos = data.ALL_ASCII_LETTERS.find(c)
        res[idx][0][pos] = 1
    return res


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, x_input, hidden):
        combined = torch.cat([x_input, hidden], dim=1)
        new_hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, new_hidden

    def new_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))


class LibRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LibRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.rnn = nn.RNN(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, x_input, hidden):
        output, out_hidden = self.rnn.forward(x_input, hidden)
        v_output = self.out(out_hidden[0])
        v_output = self.softmax(v_output)
        return v_output, out_hidden

    def new_hidden(self):
        return None


def class_from_output(output):
    assert isinstance(output, Variable)
    top_v, top_i = output.data.topk(1)
    return top_i[0][0]


def random_sample(classes, data_dict):
    assert isinstance(classes, list)
    assert isinstance(data_dict, dict)

    class_index = random.randrange(0, len(classes))
    rand_class = classes[class_index]
    name = random.choice(data_dict[rand_class])
    v_name = Variable(name_to_tensor(name))
    v_class = Variable(torch.LongTensor([class_index]))
    return rand_class, name, v_class, v_name


def convert_name(name):
    return Variable(name_to_tensor(name))


def convert_classes(classes_indices):
    return Variable(torch.LongTensor(classes_indices))


def convert_sample(name, class_idx):
    assert isinstance(name, str)
    assert isinstance(class_idx, int)
    v_name = Variable(name_to_tensor(name))
    v_class = Variable(torch.LongTensor([class_idx]))
    return v_name, v_class


def convert_batch(batch, cuda=False):
    """
    Convert batch of samples (list of (name, class_idx)) to form ready to train
    :param batch: batch of samples
    :return: tuple of (packed_sequence, classes_idx)
    """
    # sort batch according to cudnn RNN requirements
    batch.sort(key=lambda t: len(t[0]), reverse=True)
    names, classes = zip(*batch)

    lengths = list(map(len, names))
    # create padded tensor with names encoding
    padded_batch = torch.zeros(len(batch), lengths[0], INPUT_SIZE)
    for name_idx, name in enumerate(names):
        for char_idx, c in enumerate(name):
            idx = LETTERS_INDICES[c]
            padded_batch[name_idx][char_idx][idx] = 1.0

    v_data = Variable(padded_batch)
    v_classes = Variable(torch.LongTensor(classes))
    if cuda:
        v_data = v_data.cuda()
        v_classes = v_classes.cuda()
    packed_seq = rnn_utils.pack_padded_sequence(v_data, lengths, batch_first=True)

    return packed_seq, v_classes


def convert_output(rnn_output):
    """
    Unpack packed sequence from RNN back into tensor with neurons' outputs
    :param rnn_output: PackedSequence
    :return:
    """
    assert isinstance(rnn_output, rnn_utils.PackedSequence)

    v_padded, lengths = rnn_utils.pad_packed_sequence(rnn_output, batch_first=True)
    # TODO: have no idea how to make it using slices/indexing :(
    res = []
    for idx, l in enumerate(lengths):
        res.append(v_padded[idx][l-1])
#    print(res[0])
    return torch.stack(res)


def test_model(model, test_data, cuda=False):
    """
    Perform test on test dataset and return mean log loss
    :param model: rnn model to test
    :param test_data: list of test samples (name, class_idx)
    :return:
    """
    loss = 0.0
    nll_loss = nn.NLLLoss()

    for entry in test_data:
        packed_sample, classes = convert_batch([entry], cuda=cuda)
        v_output, _ = model.forward(packed_sample, None)
        loss += nll_loss(v_output, classes).data.cpu().numpy()[0]
    return loss / len(test_data)

pass
