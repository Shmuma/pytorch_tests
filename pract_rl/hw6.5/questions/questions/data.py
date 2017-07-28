import random
import pandas as pd
import numpy as np

from nltk.tokenize import TweetTokenizer

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


END_TOKEN = '<end>'


def read_embeddings(file_name, train_tokens=None):
    """
    Read embeddings file
    :param file_name: file to read
    :param train_tokens: list of list of string with sequences to filter embeddings. If None, we'll read all embeddings
    :return: tuple of dict word->id and numpy matrix with embeddings
    """
    words = {}
    vectors = []
    if train_tokens is None:
        whitelist = None
    else:
        whitelist = set()
        for tokens in train_tokens:
            whitelist.update(tokens)

    with open(file_name, "rt", encoding='utf-8') as fd:
        for l in fd:
            word, rest = l.split(' ', maxsplit=1)
            if whitelist is None or word in whitelist:
                rest = rest.split(' ')
                words[word] = len(words)
                vectors.append(list(map(float, rest)))
        words[END_TOKEN] = len(words)
        vectors.append(np.zeros((len(vectors[0], ))))

    return words, np.array(vectors)


def read_questions(file_name):
    df = pd.read_csv(file_name)
    df = pd.concat([df.question1, df.question2])
    questions = list(set(df))
    questions = filter(lambda x: type(x) is str, questions)
    return list(questions)


def tokenise_data(data):
    """
    Perform tokenisation and preprocessing of input questions
    :param data: list of strings
    :return: list of list of strings 
    """
    token = TweetTokenizer(preserve_case=False)
    return [token.tokenize(s) + [END_TOKEN] for s in data]


def tokens_to_embeddings(tokens_data, words_dict):
    """
    Convert list of list of tokens to list of indices, filtering unknown words 
    :param tokens_data: list of list of string 
    :param words_dict: map from word to their index
    :return: list of list of int 
    """
    result = []
    for tokens in tokens_data:
        indices = [words_dict[token] for token in tokens if token in words_dict]
        result.append(indices)
    return result


def batch_to_train(batch, words_dict, cuda=False):
    assert isinstance(batch, list)

    batch.sort(key=len, reverse=True)
    lens = list(map(len, batch))
    end_idx = words_dict[END_TOKEN]

    data = []
    output = []
    for sample in batch:
        data.append(sample + [end_idx] * (lens[0] - len(sample)))
        output.append(sample[1:] + [end_idx] * (lens[0] - len(sample) + 1))
    data_v = Variable(torch.LongTensor(data))
    out_v = Variable(torch.LongTensor(output))

    if cuda:
        data_v = data_v.cuda()
        out_v = out_v.cuda()

    input_seq = rnn_utils.pack_padded_sequence(data_v, lens, batch_first=True)
    output_seq = rnn_utils.pack_padded_sequence(out_v, lens, batch_first=True)

    return input_seq, output_seq.data


def iterate_batches(data, batch_tokens, shuffle=True):
    """
    Iterate over batches of data, last batch can be incomplete
    :param data: list of data samples
    :param batch_tokens: count of tokens to sample
    :param shuffle: do we need to shuffle data before iteration
    :return: yields data in chunks
    """
    assert isinstance(data, list)
    assert isinstance(batch_tokens, int)

    if shuffle:
        random.shuffle(data)
    batch = []
    batch_size = 0
    for sample in data:
        batch.append(sample)
        batch_size += len(sample)
        if batch_size > batch_tokens:
            yield batch
            batch = []
            batch_size = 0
