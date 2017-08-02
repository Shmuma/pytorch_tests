#!/usr/bin/env python3
import os
import sys
import time
import logging
import datetime
import random
import argparse
import itertools
from tqdm import tqdm
import numpy as np

from questions import data, model, plots

import torch
import torch.nn as nn
import torch.optim as optim


log = logging.getLogger("train")


TRAIN_DATA_FILE = "~/work/data/experiments/quora-questions/train.csv"
GLOVE_EMBEDDINGS = "~/work/data/experiments/glove.6B.50d.txt"
HIDDEN_SIZE = 512
EPOCHES = 500
BATCH_TOKENS = 50000
#BATCH_TOKENS = 16

# H_SOFTMAX = True


if __name__ == "__main__":
    random.seed(1234)
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)-14s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embeddings", default=GLOVE_EMBEDDINGS,
                        help="File with embeddings to read, default=%s" % GLOVE_EMBEDDINGS)
    parser.add_argument("-t", "--train", default=TRAIN_DATA_FILE,
                        help="Train data file, default=%s" % TRAIN_DATA_FILE)
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda")
    parser.add_argument("--name", required=True, help="Name of directory to save models to")
    parser.add_argument("--tiny", default=False, action='store_true', help="Limit amount of samples to 5000")
    args = parser.parse_args()

    save_path = os.path.join("saves", args.name)
    if args.tiny:
        save_path += "-tiny"
    os.makedirs(save_path, exist_ok=args.tiny)

    time_s = time.time()
    log.info("Reading training data from %s", args.train)
    train = data.read_questions(os.path.expanduser(args.train))
    train.sort()
    random.shuffle(train)
    if args.tiny:
        train = train[:5000]
    log.info("Done, got %d input sequences", len(train))

    log.info("Tokenize questions...")
    train_tokens = data.tokenise_data(train)
    total_tokens = sum(map(len, train_tokens))
    log.info("Done, have %d total tokens", total_tokens)

    log.info("Reading embeddings from %s", args.embeddings)
    words, embeddings = data.read_embeddings(os.path.expanduser(args.embeddings), train_tokens)
    rev_words = {idx: token for token, idx in words.items()}
    log.info("Read successfully, shape = %s", embeddings.shape)

    log.info("Convert samples to index lists...")
    train_sequences = data.tokens_to_embeddings(train_tokens, words)
    log.info("Done, sample: %s -> %s", train[0], train_sequences[0])
    log.info("All preparations took %s", datetime.timedelta(seconds=time.time() - time_s))

    net = model.FixedEmbeddingsModel(embeddings, HIDDEN_SIZE)
#    net_map = model.SoftmaxMappingModule(HIDDEN_SIZE, len(embeddings))
    net_map = model.TwoLevelSoftmaxMappingModule(HIDDEN_SIZE, len(embeddings), freq_ratio=0.005, class_size_mul=2.0)
    if args.cuda:
        net.cuda()
        net_map.cuda()
    lr = 0.001
    lr_decay = 0.9
    lr_decay_epoches = 50

    def make_optimizer(opt_lr):
        return optim.Adam(filter(lambda p: p.requires_grad, itertools.chain(net.parameters(), net_map.parameters())),
                          lr=opt_lr)

    optimizer = make_optimizer(lr)

    best_loss = None
    epoch_losses = []

    for epoch in range(EPOCHES):
        time_s = time.time()
        losses = []

        for batch in tqdm(data.iterate_batches(train_sequences, BATCH_TOKENS),
                          total=total_tokens // BATCH_TOKENS):
            net.zero_grad()
            net_map.zero_grad()
            input_seq, valid_v = data.batch_to_train(batch, words, args.cuda)
            out, _ = net(input_seq)
            # output is PackedSequence, so, we need to pass .data field
            loss_v = net_map(out.data, valid_v)
            loss_v.backward()
            optimizer.step()
            losses.append(loss_v.cpu().data[0])
        speed = len(train_sequences) / (time.time() - time_s)
        loss = np.mean(losses)
        epoch_losses.append(loss)

        log.info("Epoch %d: mean_loss=%.4f, speed=%.3f item/s", epoch, loss, speed)
        # question = model.generate_question(net, net_map, words, rev_words, args.cuda)
        # print("Question on epoch %d: %s" % (epoch, " ".join(question)))
        # sys.stdout.flush()
        plots.plot_progress(epoch_losses, os.path.join(save_path, "status.html"))

        if best_loss is None or best_loss > loss:
            path = os.path.join(save_path, "%04d-model-loss=%.4f.data" % (epoch, loss))
            new_state = net.state_dict()
            torch.save(new_state, path)
            log.info("Best loss updated: %.4f -> %.4f, model saved in %s",
                     np.inf if best_loss is None else best_loss, loss, path)
            best_loss = loss
        # lr decay
        if lr_decay is not None and epoch > 0 and epoch % lr_decay_epoches == 0:
            lr *= lr_decay
            optimizer = make_optimizer(lr)
            log.info("LR decreased to %.5f", lr)
    pass
