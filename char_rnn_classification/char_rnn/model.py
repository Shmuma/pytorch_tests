import torch

from . import data


def name_to_tensor(name):
    res = torch.zeros((len(name), 1, len(data.ALL_ASCII_LETTERS)))
    for idx, c in enumerate(name):
        pos = data.ALL_ASCII_LETTERS.find(c)
        res[idx][0][pos] = 1
    return res

pass
