#!/usr/bin/env python3
# Reimplementation of tutorial from https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
import argparse
import random

from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from char_rnn import data
from char_rnn import model
from char_rnn import plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda")
    parser.add_argument("--epoches", default=10, type=int, help="Count of epoches to train")
    parser.add_argument("--test", default=0.2, type=float, help="Ratio of train samples to be used for test. Default=0.2")
    parser.add_argument("--plot", help="Generate html page with plots")
    parser.add_argument("--save", help="Save model after training to this file")
    parser.add_argument("--lr-decay", type=float, help="Float multiplier to decay LR. If not specified (default), do not decay")
    args = parser.parse_args()

    data_dict = data.read_data()
    classes = list(sorted(data_dict.keys()))
    train_data, test_data = data.prepare_train_test(classes, data_dict, test_ratio=args.test)

    print("Read %d classes, we have %d train samples and %d test samples" % (len(data_dict), len(train_data),
                                                                             len(test_data)))

    rnn = model.LibRNN(input_size=model.INPUT_SIZE, hidden_size=100, output_size=len(data_dict))
#    rnn = torch.nn.RNN(input_size=model.INPUT_SIZE, hidden_size=100)
    if args.cuda:
        rnn = rnn.cuda()
    opt_target = nn.NLLLoss()
    lr = 0.005
    optimizer = optim.SGD(rnn.parameters(), lr=lr)

    loss_sum = 0.0
    train_losses = []
    test_losses = []
    lr_values = []

    for epoch in tqdm(range(args.epoches)):
        random.shuffle(train_data)
        for name, class_idx in train_data:
            rnn.zero_grad()

            v_output = v_hidden = rnn.new_hidden()
            v_name, v_class = model.convert_sample(name, class_idx)
            if args.cuda:
                v_class = v_class.cuda()
                v_name = v_name.cuda()

            for v_char in v_name:
                v_output, v_hidden = rnn.forward(v_char, v_hidden)
            loss = opt_target(v_output, v_class)
            loss.backward()
            optimizer.step()

            loss_sum += loss.data[0]
        loss_sum /= len(train_data)
        test_loss = model.test_model(rnn, test_data, cuda=args.cuda)
        print("%d: loss=%.5f, test_loss=%.5f, lr=%.5f" % (epoch, loss_sum, test_loss, lr))
        train_losses.append((epoch, loss_sum))
        test_losses.append((epoch, test_loss))
        if args.lr_decay is not None:
            lr *= args.lr_decay
            optimizer = optim.SGD(rnn.parameters(), lr=lr)
        lr_values.append((epoch, lr))

        loss_sum = 0.0
        if args.plot is not None:
            plots.plot_data(train_losses, test_losses, lr_values, args.plot)
    if args.save is not None:
        torch.save(rnn.state_dict(), args.save)
    pass
