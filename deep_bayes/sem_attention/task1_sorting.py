#!/usr/bin/env python3
"""
Implementation of sorting network with attention.

Original tutorial is on lasagna: 
https://github.com/bayesgroup/deepbayes2017/blob/master/sem3-attention/Attention_seminar%20(Start%20here).ipynb
"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


CODE_SIZE = 10


def generate_sample(min_length=3, max_length=10, code_size=CODE_SIZE):
    assert code_size >= max_length
    length = np.random.randint(min_length, max_length)

    keys = np.random.permutation(length)
    values = np.random.permutation(length)
    input_pairs = list(zip(keys, values))

    input_1hot = np.zeros([length + 1, code_size * 2])
    for i, (k, v) in enumerate(input_pairs):
        input_1hot[i + 1][k] = 1
        input_1hot[i + 1][code_size + v] = 1

    sorted_pairs = sorted(input_pairs, key=lambda p: p[0])

    target_1hot = np.zeros([length + 1, code_size * 2])
    for i, (k, v) in enumerate(sorted_pairs):
        target_1hot[i + 1][k] = 1
        target_1hot[i + 1][code_size + v] = 1

    return input_1hot, target_1hot


def test_sample():
    inp, out = generate_sample(max_length=5, code_size=5)
    print ('-' * 9 + "KEY" + '-' * 9 + ' ' + '+' * 9 + "VAL" + "+" * 9)
    print ("Input pairs:\n", inp)
    print ("Target pairs:\n", out)


if __name__ == "__main__":
    test_sample()
    pass
