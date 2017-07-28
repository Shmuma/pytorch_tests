#!/usr/bin/env python3
import os
import logging
import argparse

from questions import data, model

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable


log = logging.getLogger("train")

TRAIN_DATA_FILE = "~/work/data/experiments/quora-questions/train.csv"
GLOVE_EMBEDDINGS = "~/work/data/experiments/glove.6B.50d.txt"
HIDDEN_SIZE = 128


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)-14s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embeddings", default=GLOVE_EMBEDDINGS,
                        help="File with embeddings to read, default=%s" % GLOVE_EMBEDDINGS)
    parser.add_argument("-t", "--train", default=TRAIN_DATA_FILE,
                        help="Train data file, default=%s" % TRAIN_DATA_FILE)
    args = parser.parse_args()

    log.info("Reading embeddings from %s", args.embeddings)
    words, embeddings = data.read_embeddings(os.path.expanduser(args.embeddings))
    log.info("Read successfully, shape = %s", embeddings.shape)

    log.info("Reading training data from %s", args.train)
    train = data.read_questions(os.path.expanduser(args.train))
    log.info("Done, got %d input sequences", len(train))

    log.info("Tokenize samples...")
    train_sequences, unknown_words = data.tokenize_questions(train, words)
    log.info("Done, sample: %s -> %s", train[0], train_sequences[0])
    log.info("There are %d unknown words", len(unknown_words))

    net = model.FixedEmbeddingsModel(embeddings, HIDDEN_SIZE)

    batch = [train_sequences[0]]
    input_seq, valid_t = data.batch_to_train(batch, words)

    out, h = net(input_seq)
    print(out)

    pass
