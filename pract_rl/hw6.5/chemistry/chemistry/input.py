import random
import pandas as pd
import numpy as np

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


FILE_NAME = "molecules.tsv.gz"
END_TOKEN = 'END'


def read_data(file_name=FILE_NAME):
    df = pd.read_csv(file_name, sep='\t')
    is_str = lambda s: type(s) is str
    df = df[df.molecular_formula.apply(is_str) & df.common_name.apply(is_str)]
    filter_fn = lambda s: s.replace("\n", "")
    return list(zip(df.molecular_formula, map(filter_fn, df.common_name)))


def split_train_test(data, ratio=0.8):
    train_len = int(len(data) * ratio)
    return data[:train_len], data[train_len:]


class Vocabulary:
    """
    Vocabulary which used to encode and decode sequences, built dynamically from datas
    """
    def __init__(self, data, extra_items=()):
        r = set(list("".join(data)))

        self.tokens = list(sorted(r))
        self.tokens.extend(extra_items)
        self.token_index = {token: idx for idx, token in enumerate(self.tokens)}
        self.index_token = {idx: token for idx, token in enumerate(self.tokens)}

    def __repr__(self):
        return "Vocabulary(tokens_len=%d)" % len(self.tokens)

    def __len__(self):
        return len(self.tokens)


def encode_batch(input_batch, vocab, cuda=False):
    """
    Encode input batch
    :param input_batch: list of tuples with input and output sequences
    :param vocab: Vocabulary instance to use
    :return: tuple of PackedSequence and output sequence
    """
    assert isinstance(input_batch, list)
    assert isinstance(vocab, Vocabulary)

    input_batch.sort(key=lambda p: len(p[0]), reverse=True)
    input_data, output_data = zip(*input_batch)
    lens = list(map(len, input_data))
    # padded sequence has shape of batch*longest seq*embedding
    padded = np.zeros(shape=(len(input_batch), lens[0], len(vocab)), dtype=np.float32)
    for sample_idx, sample in enumerate(input_data):
        for token_idx, token in enumerate(sample):
            padded[sample_idx][token_idx][vocab.token_index[token]] = 1.0

    # pack sequence
    padded_v = Variable(torch.from_numpy(padded))
    if cuda:
        padded_v = padded_v.cuda()
    input_packed = rnn_utils.pack_padded_sequence(padded_v, lens, batch_first=True)
    return input_packed, output_data


def iterate_batches(data, batch_size):
    random.shuffle(data)
    ofs = 0
    while ofs < len(data):
        yield data[ofs:ofs+batch_size]
        ofs += batch_size
