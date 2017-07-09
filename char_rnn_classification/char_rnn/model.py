import random

import torch
import torch.nn as nn
from torch.autograd import Variable

from . import data

INPUT_SIZE = len(data.ALL_ASCII_LETTERS)


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


pass
