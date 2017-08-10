#!/usr/bin/env python3
import logging
import random

from chemistry import input
from chemistry import model


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
    print(hid.size())
    pass
