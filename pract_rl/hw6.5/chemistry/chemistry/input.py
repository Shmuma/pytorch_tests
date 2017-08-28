import random
import string
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


def generate_dataset(dataset_kind, samples_count=10000, input_size=10):
    """
    Generate synthetic dataset for verification. Currently those kinds are supported:
    1. first_last: generate random word, output should be first and last letter
    2. count: by random sequence and digit repeat digit given amount of times
    :param dataset_kind:
    :return: list of tuples
    """
    chars = list(string.ascii_lowercase)
    if dataset_kind == 'first_last':
        input_data = ["".join(np.random.choice(chars, input_size, replace=True).tolist()) for _ in range(samples_count)]
        result = [(s, s[0] + s[-1]) for s in input_data]
        return result
    elif dataset_kind == 'count':
        input_data = ["".join(np.random.choice(chars, 3, replace=True).tolist()) for _ in range(samples_count)]
        result = []
        for s in input_data:
            c = np.random.choice(range(10))
            result.append(("%s%d" % (s, c), s * c))
        return result
    else:
        raise ValueError


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


def encode_batch(input_batch, vocab, cuda=False, volatile=False):
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
    padded_v = Variable(torch.from_numpy(padded), volatile=volatile)
    if cuda:
        padded_v = padded_v.cuda()
    input_packed = rnn_utils.pack_padded_sequence(padded_v, lens, batch_first=True)
    return input_packed, list(output_data)


def split_batches(data, batch_tokens, max_sequences=None):
    """
    Return chunk bounds for batches
    :param data: list of input,output sequences
    :param batch_tokens: limit of tokens in the batch
    :return: list of tuples with start:end bounds of batch
    """
    res = []
    tokens = 0
    batch_start = 0
    batch_len = 0
    while batch_start + batch_len < len(data):
        if tokens >= batch_tokens or (max_sequences is not None and batch_len >= max_sequences):
            res.append((batch_start, batch_start+batch_len))
            batch_start += batch_len
            batch_len = 0
            tokens = 0
        tokens += len(data[batch_start+batch_len][0]) + len(data[batch_start+batch_len][1])
        batch_len += 1
    if tokens > 0:
        res.append((batch_start, batch_start+batch_len))
    return res


def encode_output_batch(sequences, output_vocab, cuda=False):
    """
    Encode list of sequences into decoder's input and create valid indices sequences
    :param sequences: list of sorted sequences
    :param output_vocab:
    :return: Tuple with PackedSequence, Variable with valid indices
    """
    assert isinstance(sequences, list)
    assert isinstance(output_vocab, Vocabulary)

    end_token_idx = output_vocab.token_index[END_TOKEN]
    lens = list(map(len, sequences))
    input_emb = np.zeros(shape=(len(sequences), lens[0], len(output_vocab)), dtype=np.float32)
    output_indices = np.full(shape=(len(sequences), lens[0]), fill_value=end_token_idx, dtype=np.int64)
    input_emb[:, :, end_token_idx] = 1.0
    for seq_ofs, sequence in enumerate(sequences):
        for token_ofs, token in enumerate(sequence):
            token_idx = output_vocab.token_index[token]
            input_emb[seq_ofs, token_ofs, token_idx] = 1.0
            if token_ofs > 0:
                output_indices[seq_ofs, token_ofs-1] = token_idx
    input_v = Variable(torch.from_numpy(input_emb))
    output_v = Variable(torch.from_numpy(output_indices))
    if cuda:
        input_v = input_v.cuda()
        output_v = output_v.cuda()
    input_packed = rnn_utils.pack_padded_sequence(input_v, lens, batch_first=True)
    output_packed = rnn_utils.pack_padded_sequence(output_v, lens, batch_first=True)
    return input_packed, output_packed.data
