#!/usr/bin/env python3
import os
import time
import logging
import datetime
import argparse
from tqdm import tqdm
import numpy as np

from questions import data, model

import torch
import torch.nn as nn
import torch.optim as optim


log = logging.getLogger("train")


TRAIN_DATA_FILE = "~/work/data/experiments/quora-questions/train.csv"
GLOVE_EMBEDDINGS = "~/work/data/experiments/glove.6B.50d.txt"
HIDDEN_SIZE = 128
EPOCHES = 100
BATCH_SIZE = 256


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)-14s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embeddings", default=GLOVE_EMBEDDINGS,
                        help="File with embeddings to read, default=%s" % GLOVE_EMBEDDINGS)
    parser.add_argument("-t", "--train", default=TRAIN_DATA_FILE,
                        help="Train data file, default=%s" % TRAIN_DATA_FILE)
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda")
    parser.add_argument("--name", required=True, help="Name of directory to save models to")
    args = parser.parse_args()

    save_path = os.path.join("saves", args.name)
    os.makedirs(save_path, exist_ok=False)

    time_s = time.time()
    log.info("Reading training data from %s", args.train)
    train = data.read_questions(os.path.expanduser(args.train))
    log.info("Done, got %d input sequences", len(train))

    log.info("Tokenize questions...")
    train_tokens = data.tokenise_data(train)
    log.info("Done, have %d total tokens", sum(map(len, train_tokens)))

    log.info("Reading embeddings from %s", args.embeddings)
    words, embeddings = data.read_embeddings(os.path.expanduser(args.embeddings), train_tokens)
    log.info("Read successfully, shape = %s", embeddings.shape)

    log.info("Convert samples to index lists...")
    train_sequences = data.tokens_to_embeddings(train_tokens, words)
    log.info("Done, sample: %s -> %s", train[0], train_sequences[0])
    log.info("All preparations took %s", datetime.timedelta(seconds=time.time() - time_s))

    net = model.FixedEmbeddingsModel(embeddings, HIDDEN_SIZE)
    if args.cuda:
        net.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01)
    objective = nn.CrossEntropyLoss()

    best_loss = None

    for epoch in range(EPOCHES):
        time_s = time.time()
        losses = []

        for batch in tqdm(data.iterate_batches(train_sequences, BATCH_SIZE),
                          total=len(train_sequences)//BATCH_SIZE):
            net.zero_grad()
            input_seq, valid_v = data.batch_to_train(batch, words, args.cuda)
            out, _ = net(input_seq)

            loss_v = objective(out, valid_v)
            loss_v.backward()
            optimizer.step()
            losses.append(loss_v.cpu().data[0])

        speed = len(losses) * BATCH_SIZE / (time.time() - time_s)
        loss = np.mean(losses)
        log.info("Epoch %d: mean_loss=%.4f, speed=%.3f item/s", epoch, loss, speed)

        if best_loss is None or best_loss > loss:
            path = os.path.join(save_path, "%04d-model-loss=%.4f.data" % (epoch, loss))
            torch.save(net.state_dict(), path)
            log.info("Best loss updated: %.4f -> %.4f, model saved in %s",
                     np.inf if best_loss is None else best_loss, loss, path)
            best_loss = loss

    pass
