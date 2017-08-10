#!/usr/bin/env python3
import logging
import random
import numpy as np

from chemistry import input
from chemistry import model

import torch
import torch.nn as nn
import torch.nn.functional as nn_func
import torch.optim as optim
from torch.autograd import Variable


SEED = 2345  # obtained from fair dice roll, do not change!
HIDDEN_SIZE = 128


log = logging.getLogger("train")


if __name__ == "__main__":
    random.seed(SEED)
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)-14s %(message)s", level=logging.INFO)

    # read dataset
    data = input.read_data()
    random.shuffle(data)
    log.info("Have %d samples", len(data[0]))

    input_vocab  = input.Vocabulary(map(lambda p: p[0], data))
    output_vocab = input.Vocabulary(map(lambda p: p[1], data), extra_items=(input.END_TOKEN, ))

    log.info("Input: %s", input_vocab)
    log.info("Output: %s", output_vocab)

    # split into train/test
    train_data, test_data = input.split_train_test(data, ratio=0.9)
    log.info("Train has %d items, test %d", len(train_data), len(test_data))

    # train
    encoder = model.Encoder(input_size=len(input_vocab), hidden_size=HIDDEN_SIZE)
    decoder = model.Decoder(hidden_size=HIDDEN_SIZE, output_size=len(output_vocab))

    input_packed, output_seq = input.encode_batch(train_data[:1], input_vocab)
    out, hid = encoder(input_packed)
    next_token = input.END_TOKEN

    print(hid.size())
    generated = []

    for valid_token in list(output_seq[0]) + [input.END_TOKEN]:
        # input to the decoder
        token_emb = np.zeros(shape=(1, len(output_vocab)), dtype=np.float32)
        token_emb[0][output_vocab.token_index[next_token]] = 1.0
        token_emb_v = Variable(torch.from_numpy(token_emb))
        dec_out, dec_hid = decoder(token_emb_v, hid)
        # find next tokens for batch
        indices = torch.multinomial(nn_func.softmax(dec_out), num_samples=1)
        next_token = output_vocab.index_token[indices.cpu().data.numpy()[0][0]]
        generated.append(next_token)
        loss = nn_func.cross_entropy(dec_out, Variable(torch.LongTensor([output_vocab.token_index[valid_token]])))
        print(dec_hid.size())
        break
    pass
