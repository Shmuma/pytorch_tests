#!/usr/bin/env python3
# Reimplementation of tutorial from https://github.com/spro/practical-pytorch/tree/master/char-rnn-classification

from char_rnn import data


if __name__ == "__main__":
    train_data = data.read_data()
    print("Read %d classes" % len(train_data))
    pass
