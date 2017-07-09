#!/usr/bin/env python3
# Reimplementation of tutorial from https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification
import argparse

from tqdm import tqdm
from torch import nn
from torch import optim

from char_rnn import data
from char_rnn import model

STEPS = 1000000
PRINT_EVERY_STEP = 10000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda")
    args = parser.parse_args()

    train_data = data.read_data()
    print("Read %d classes" % len(train_data))

    classes = list(sorted(train_data.keys()))

    rnn = model.RNN(input_size=model.INPUT_SIZE, hidden_size=100, output_size=len(train_data))
    if args.cuda:
        rnn = rnn.cuda()
    opt_target = nn.NLLLoss()
    optimizer = optim.SGD(rnn.parameters(), lr=0.005)

    loss_sum = 0.0
    losses = []

    for step_idx in tqdm(range(1, STEPS+1)):
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
        if step_idx % PRINT_EVERY_STEP == 0:
            loss_sum /= PRINT_EVERY_STEP
            print("%d: loss=%.5f" % (step_idx, loss_sum))
            losses.append(loss_sum)
            loss_sum = 0.0

    pass
