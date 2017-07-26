#!/usr/bin/env python3
import time
import numpy as np
import logging
import argparse

import torch.nn as nn
from torch import optim

from lib import input
from lib import model


HIDDEN_SIZE = 128
EPOCHES = 200
BATCH_SIZE = 16

log = logging.getLogger("train")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)-14s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help="Enable cuda")
    args = parser.parse_args()

    data = input.read_data()
    input_encoder = input.InputEncoder(data)

    log.info("Read %d train samples, encoder len=%d, first 10: %s", len(data),
             len(input_encoder), ', '.join(data[:10]))

    net = model.Model(len(input_encoder), hidden_size=HIDDEN_SIZE)
    if args.cuda:
        net = net.cuda()
    objective = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=0.001)

    for epoch in range(EPOCHES):
        losses = []
        start_ts = time.time()

        for batch in input.iterate_batches(data, BATCH_SIZE):
            net.zero_grad()
            packed_seq, true_indices = input.batch_to_train(batch, input_encoder, cuda=args.cuda)
            out, _ = net(packed_seq)
            v_loss = objective(out, true_indices)
            v_loss.backward()
            losses.append(v_loss.data[0])
            optimizer.step()

        speed = len(losses) * BATCH_SIZE / (time.time() - start_ts)
        gen_names = [model.generate_name(net, input_encoder, cuda=args.cuda) for _ in range(10)]
        log.info("Epoch %d: mean_loss=%.4f, speed=%.3f item/s, names: %s", epoch, np.mean(losses), speed,
                 ", ".join(gen_names))

    pass
