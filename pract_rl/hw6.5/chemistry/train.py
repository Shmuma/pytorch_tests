#!/usr/bin/env python3
import os
import time
import argparse
import collections
import itertools
import logging
import random
import numpy as np
from tqdm import tqdm

from chemistry import input
from chemistry import model
from chemistry import plots

import torch
import torch.nn.functional as nn_func
import torch.optim as optim
from torch.autograd import Variable


SEED = 2345  # obtained from fair dice roll, do not change!
HIDDEN_SIZE = 128

EPOCHES = 100
BATCH_SIZE = 200


log = logging.getLogger("train")


if __name__ == "__main__":
    random.seed(SEED)
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)-14s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda mode")
    parser.add_argument("--name", required=True, help="Name of directory to save models to")
    parser.add_argument("--tiny", default=False, action='store_true', help="Limit amount of samples to 5000")
    args = parser.parse_args()

    save_path = os.path.join("saves", args.name)
    if args.tiny:
        save_path += "-tiny"
    os.makedirs(save_path, exist_ok=args.tiny)

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
    if args.tiny:
        train_data = train_data[:5000]
    random.shuffle(train_data)
    log.info("Train has %d items, test %d", len(train_data), len(test_data))

    # train
    encoder = model.Encoder(input_size=len(input_vocab), hidden_size=HIDDEN_SIZE)
    decoder = model.Decoder(hidden_size=HIDDEN_SIZE, output_size=len(output_vocab))
    if args.cuda:
        encoder.cuda()
        decoder.cuda()
    optimizer = optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.01)

    end_token_idx = output_vocab.token_index[input.END_TOKEN]
    epoch_losses = []

    for epoch in range(EPOCHES):
        losses = []
        time_counters = collections.Counter()
        ts_epoch = time.time()

        for batch in tqdm(input.iterate_batches(train_data, BATCH_SIZE), total=len(train_data) / BATCH_SIZE):
            ts = time.time()
            optimizer.zero_grad()
            input_packed, output_sequences = input.encode_batch(batch, input_vocab, cuda=args.cuda)
            time_counters['prepare_batch'] += time.time() - ts

            ts = time.time()
            hid = encoder(input_packed)
            time_counters['encoder'] += time.time() - ts

            # input for decoder
            ts = time.time()
            input_token_indices = torch.LongTensor([[end_token_idx]] * len(batch))
            max_out_len = max(map(len, output_sequences)) + 1 # extra item for final END_TOKEN
            # expected tokens
            valid_token_indices = np.full(shape=(len(batch), max_out_len), fill_value=end_token_idx, dtype=np.int64)
            for idx, out_seq in enumerate(output_sequences):
                valid_token_indices[idx][:len(out_seq)] = [output_vocab.token_index[c] for c in out_seq]
            valid_token_indices_v = Variable(torch.from_numpy(valid_token_indices.transpose()))
            time_counters['expected_tokens'] += time.time() - ts

            input_emb = torch.FloatTensor(len(batch), len(output_vocab))
            input_emb_v = Variable(input_emb)
            if args.cuda:
                input_emb_v = input_emb_v.cuda()
                input_token_indices = input_token_indices.cuda()
                valid_token_indices_v = valid_token_indices_v.cuda()

            # iterate decoder over largest sequence
            ts = time.time()
            for ofs in range(max_out_len):
                input_emb_v.data.zero_()
                input_emb_v.data.scatter_(1, input_token_indices, 1.0)

                # on first iteration pass hidden from encoder
                dec_out, hid = decoder(input_emb_v, hid)
                # sample next tokens for decoder's input
                input_token_indices = torch.multinomial(nn_func.softmax(dec_out), num_samples=1).data
                loss = nn_func.cross_entropy(dec_out, valid_token_indices_v[ofs])
                loss.backward(retain_variables=True)
                losses.append(loss.cpu().data.numpy())
            time_counters['decoder'] += time.time() - ts
            optimizer.step()
        ts_epoch = time.time() - ts_epoch
        log.info("Epoch %d: mean_loss=%.4f", epoch, np.mean(losses))
        epoch_losses.append(np.mean(losses))
        plots.plot_progress(epoch_losses, os.path.join(save_path, "status.html"))
        log.info("Timings:")
        for name, ts in time_counters.most_common():
            log.info("  %s: %.3f seconds, %.2f%%", name, ts, 100.0 * ts / ts_epoch)
    pass
