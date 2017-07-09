#!/usr/bin/env python3
import argparse

import torch
from torch.autograd import Variable

from char_rnn import data
from char_rnn import model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", required=True, help="Name of the model to load")
    parser.add_argument("--hidden", default=100, type=int, help="Count of intput units used to train")
    args = parser.parse_args()

    data_dict = data.read_data()
    classes = list(sorted(data_dict.keys()))

    rnn = model.RNN(len(data.ALL_ASCII_LETTERS), args.hidden, len(classes))
    rnn.load_state_dict(torch.load(args.load))

    print("Model loaded, please enter names")

    while True:
        line = input("> ")
        line = data.name_to_ascii(line)
        v_name = Variable(model.name_to_tensor(line))
        v_hidden = rnn.new_hidden()
        v_output = None
        for v_char in v_name:
            v_output, v_hidden = rnn.forward(v_char, v_hidden)
        top_v, top_i = v_output.data.topk(3)

        for score, idx in zip(top_v.numpy().tolist()[0], top_i.numpy().tolist()[0]):
            print("(%.2f) %s" % (score, classes[idx]))
    pass
