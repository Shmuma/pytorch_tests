import random
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils


TRAIN_FILE = "names"


def read_data(file_name=TRAIN_FILE):
    with open(file_name, "rt", encoding='utf-8') as fd:
        return list(map(lambda s: s.strip(), fd.readlines()))


class InputEncoder:
    START_TOKEN = '\0'
    END_TOKEN = '\1'

    def __init__(self, data):
        self.alphabet = list(sorted(set("".join(data) + self.END_TOKEN + self.START_TOKEN)))
        self.idx = {c: pos for pos, c in enumerate(self.alphabet)}

    def __len__(self):
        return len(self.alphabet)

    def encode(self, s, width=None, raw=False):
        assert isinstance(s, str)

        if width is None:
            width = len(s)
        if not raw:
            width += 2
        res = np.zeros((width, len(self)), dtype=np.float32)
        in_s = s if raw else self.START_TOKEN + s + self.END_TOKEN
        for sample_idx, c in enumerate(in_s):
            idx = self.idx[c]
            res[sample_idx][idx] = 1.0

        return res

    def end_token(self):
        """
        Get vector for end token, it will be used for padding of sequences (as we need to predict end token at the end)
        :return: numpy array
        """
        r = np.zeros((len(self),), dtype=np.float32)
        r[self.idx[self.END_TOKEN]] = 1.0
        return r

    def chars_to_indices(self, s):
        return [self.idx[c] for c in s]

    def indices_to_chars(self, indices):
        return [self.alphabet[idx] for idx in indices]

    def char_to_index(self, c):
        return self.idx[c]


def iterate_batches(data, batch_size, shuffle=True):
    """
    Iterate over batches of data, last batch can be incomplete
    :param data: list of data samples
    :param batch_size: length of batch to sample
    :param shuffle: do we need to shuffle data before iteration
    :return: yields data in chunks
    """
    assert isinstance(data, list)
    assert isinstance(batch_size, int)

    if shuffle:
        random.shuffle(data)
    ofs = 0
    while ofs*batch_size < len(data):
        yield data[ofs*batch_size:(ofs+1)*batch_size]
        ofs += 1


def batch_to_train(batch, encoder, cuda=False):
    """
    Convert batch into train input
    :param batch: list with batch samples
    :param encoder: InputEncoder instance
    :return: packed_sequence, next_characters
    """
    assert isinstance(batch, list)
    assert isinstance(encoder, InputEncoder)

    # this order is requirement of CuDNN RNN functions
    batch.sort(key=len, reverse=True)
    lens = list(map(lambda s: len(s)+2, batch)) # 2 is for start and end tokens

    max_len = lens[0]
    # resulting shape will be batch*max_sequence*embedding and filled by default with end_token encoding
    data = np.tile(encoder.end_token(), (len(batch), max_len, 1))
    for idx, sample in enumerate(batch):
        entry = encoder.encode(sample)
        data[idx][:entry.shape[0]] = entry

    v_data = Variable(torch.from_numpy(data))
    if cuda:
        v_data = v_data.cuda()
    packed_seq = rnn_utils.pack_padded_sequence(v_data, lens, batch_first=True)

    end_idx = encoder.char_to_index(encoder.END_TOKEN)
    true_idx = np.full((len(batch), max_len), fill_value=end_idx, dtype=np.long)

    for sample_idx, sample in enumerate(batch):
        indices = encoder.chars_to_indices(sample)
        true_idx[sample_idx][:len(indices)] = indices

    v_true_idx = Variable(torch.LongTensor(true_idx))
    if cuda:
        v_true_idx = v_true_idx.cuda()
    true_seq = rnn_utils.pack_padded_sequence(v_true_idx, lens, batch_first=True)

    return packed_seq, true_seq.data

