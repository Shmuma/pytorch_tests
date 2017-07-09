#!/usr/bin/env python3
# Reimplementation of tutorial from https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
import argparse

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
    parser.add_argument("--plot", help="Generate html page with plots")
    parser.add_argument("--save", help="Save model after training to this file")
    args = parser.parse_args()

    train_data = data.read_data()
    epoch_size = sum(map(len, train_data.values()))
    print("Read %d classes, total samples %d" % (len(train_data), epoch_size))

    classes = list(sorted(train_data.keys()))

    rnn = model.RNN(input_size=model.INPUT_SIZE, hidden_size=100, output_size=len(train_data))
    if args.cuda:
        rnn = rnn.cuda()
    opt_target = nn.NLLLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=0.005)

    loss_sum = 0.0
    losses = []

    for epoch in tqdm(range(args.epoches)):
        for _ in range(epoch_size):
            rnn.zero_grad()
            v_output = v_hidden = rnn.new_hidden()

            sample_class, sample_name, v_class, v_name = model.random_sample(classes, train_data)
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
        loss_sum /= epoch_size
        print("%d: loss=%.5f" % (epoch, loss_sum))
        losses.append((epoch, loss_sum))
        loss_sum = 0.0
        if args.plot is not None:
            plots.plot_data(losses, args.plot)
    if args.save is not None:
        torch.save(rnn.state_dict(), args.save)
    pass
