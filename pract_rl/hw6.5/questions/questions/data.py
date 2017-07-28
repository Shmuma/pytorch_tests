import pandas as pd
import numpy as np

from nltk.tokenize import TweetTokenizer

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


UNKNOWN_TOKEN = '<unk>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'


def read_embeddings(file_name):
    """
    Read embeddings file
    :param file_name:
    :return: tuple of dict word->id and numpy matrix with embeddings
    """
    words = {}
    vectors = []

    with open(file_name, "rt", encoding='utf-8') as fd:
        for idx, l in enumerate(fd):
            word, *rest = l.split(' ')
            words[word] = idx
            vectors.append(list(map(float, rest)))
        words[UNKNOWN_TOKEN] = len(words)
        vectors.append(np.zeros((len(vectors[0], ))))
        words[START_TOKEN] = len(words)
        vectors.append(np.random.rand(len(vectors[0])))
        words[END_TOKEN] = len(words)
        vectors.append(np.random.rand(len(vectors[0])))

    return words, np.array(vectors)


def read_questions(file_name):
    df = pd.read_csv(file_name)
    df = pd.concat([df.question1, df.question2])
    questions = list(set(df))
    questions = filter(lambda x: type(x) is str, questions)
    return list(questions)


def tokenize_questions(data, words_dict):
    """
    Perform input data tokenisation using Tweet tokeniser from nltk and convert data into word indices
    :param data: list of sentences to be tokenised
    :param words_dict: map of words -> idx
    :return: tuple of [[idx]*], set_of_unknown_words
    """
    token = TweetTokenizer()
    unknown_words = set()
    unk_idx = words_dict[UNKNOWN_TOKEN]
    start_idx = words_dict[START_TOKEN]
    end_idx = words_dict[END_TOKEN]
    res = []
    for s in data:
        items = [start_idx]
        for w in token.tokenize(s.lower()):
            idx = words_dict.get(w, unk_idx)
            items.append(idx)
            if idx == unk_idx:
                unknown_words.add(w)
        items.append(end_idx)
        res.append(items)
    return res, unknown_words


def batch_to_train(batch, words_dict):
    assert isinstance(batch, list)

    batch.sort(key=len, reverse=True)
    lens = list(map(len, batch))
    end_idx = words_dict[END_TOKEN]

    data = []
    output = []
    for sample in batch:
        data.append(sample + [end_idx] * (lens[0] - len(sample)))
        output.append(sample[1:] + [end_idx] * (lens[0] - len(sample) - 1))
    data_v = Variable(torch.LongTensor(data))
    out_v = Variable(torch.LongTensor(output))

    input_seq = rnn_utils.pack_padded_sequence(data_v, lens, batch_first=True)
    output_seq = rnn_utils.pack_padded_sequence(out_v, lens, batch_first=True)

    return input_seq, output_seq.data
