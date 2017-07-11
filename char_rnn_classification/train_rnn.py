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


BATCH_SIZE = 16


if __name__ == "__main__":
    random.seed(1234)
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
        batch_ofs = 0
        while batch_ofs < len(train_data)//BATCH_SIZE:
            rnn.zero_grad()
            batch = train_data[batch_ofs*BATCH_SIZE:(batch_ofs+1)*BATCH_SIZE]
            packed_samples, classes = model.convert_batch(batch, cuda=args.cuda)
            v_output, _ = rnn.forward(packed_samples, None)
            loss = opt_target(v_output, classes)
            loss.backward()
            optimizer.step()

            loss_sum += loss.data[0]
            batch_ofs += 1
        loss_sum /= batch_ofs
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
