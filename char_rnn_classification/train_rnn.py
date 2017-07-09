#!/usr/bin/env python3
# Reimplementation of tutorial from https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
import argparse
import random

from tqdm import tqdm

import torch
from torch import nn
from torch import optim

from char_rnn import data
from char_rnn import model
from char_rnn import plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda")
    parser.add_argument("--epoches", default=10, help="Count of epoches to train")
    parser.add_argument("--test", default=0.2, help="Ratio of train samples to be used for test. Default=0.2")
    parser.add_argument("--plot", help="Generate html page with plots")
    parser.add_argument("--save", help="Save model after training to this file")
    args = parser.parse_args()

    data_dict = data.read_data()
    classes = list(sorted(data_dict.keys()))
    train_data, test_data = data.prepare_train_test(classes, data_dict, test_ratio=args.test)

    print("Read %d classes, we have %d train samples and %d test samples" % (len(data_dict), len(train_data),
                                                                             len(test_data)))

    rnn = model.RNN(input_size=model.INPUT_SIZE, hidden_size=100, output_size=len(data_dict))
    if args.cuda:
        rnn = rnn.cuda()
    opt_target = nn.NLLLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=0.005)

    loss_sum = 0.0
    losses = []

    for epoch in tqdm(range(args.epoches)):
        random.shuffle(train_data)
        for name, class_idx in train_data:
            rnn.zero_grad()
            v_output = v_hidden = rnn.new_hidden()

            v_name, v_class = model.convert_sample(name, class_idx)
            if args.cuda:
                v_class = v_class.cuda()
                v_name = v_name.cuda()
                v_hidden = v_hidden.cuda()

            for v_char in v_name:
                v_output, v_hidden = rnn.forward(v_char, v_hidden)
            loss = opt_target(v_output, v_class)
            loss.backward()
            optimizer.step()

            loss_sum += loss.data[0]
        loss_sum /= len(train_data)
        print("%d: loss=%.5f" % (epoch, loss_sum))
        losses.append((epoch, loss_sum))
        loss_sum = 0.0
        if args.plot is not None:
            plots.plot_data(losses, args.plot)
    if args.save is not None:
        torch.save(rnn.state_dict(), args.save)
    pass
