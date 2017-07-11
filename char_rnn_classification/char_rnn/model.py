import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from . import data

INPUT_SIZE = len(data.ALL_ASCII_LETTERS)


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
        output, hidden = self.rnn.forward(x_input, hidden)
        output = self.out(output.view(-1, self.hidden_size))
        output = self.softmax(output)
        return output, hidden

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


def convert_samples(names, offset, class_indices):
    """
    Convert batch of samples at given offset into train data ready to be fed to RNN
    :param names: list of names 
    :param offset: offset within every name
    :param classes_indices: list of class indices 
    :return: None if offset is beyond every name, or tuple (v_names, v_classes, hidden_slice)
    """
    res_chars = []
    res_slice = []
    res_classes = []
    for idx, name in enumerate(names):
        if offset >= len(name):
            continue
        res_slice.append(idx)
        res_chars.append(char_to_tensor(name[offset]))
        res_classes.append(class_indices[idx])

    # we've reached the end of the sequence
    if not res_classes:
        return None

    v_classes = Variable(torch.LongTensor(res_classes))
    v_names = Variable(torch.stack(res_chars))
    print(v_names.size())


def test_model(model, test_data, cuda=False):
    """
    Perform test on test dataset and return mean log loss
    :param model: rnn model to test
    :param test_data: list of test samples (name, class_idx)
    :return:
    """
    loss = 0.0
    nll_loss = nn.NLLLoss()

    for name, class_idx in test_data:
        v_output = None
        v_hidden = model.new_hidden()
        v_name, v_class = convert_sample(name, class_idx)
        if cuda:
            v_hidden = v_hidden.cuda()
            v_name = v_name.cuda()
            v_class = v_class.cuda()
        for char_idx in range(v_name.size()[0]):
            v_output, v_hidden = model.forward(v_name[char_idx:char_idx+1], v_hidden)
        loss += nll_loss(v_output, v_class).data.numpy()[0]
    return loss / len(test_data)

pass
