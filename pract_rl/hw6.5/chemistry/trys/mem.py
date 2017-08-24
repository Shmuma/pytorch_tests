#!/usr/bin/env python3
# test memory leak on long RNN sequences
from tqdm import tqdm
import random
import numpy as np
import itertools

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as nn_func
from torch.autograd import Variable


INPUT_SIZE = 10
HIDDEN_SIZE = 512
BATCHES_TO_ITERATE = 1000
BATCH_SIZE = 200
INPUT_SEQ_LEN = 400
DECODER_HEAD_LEN = 100

# Initial results:

# * hidden_size=16:
# input_seq_len = 5: 409MB
# input_seq_len = 500: 623MB

# * hidden_size=512:
# input_seq_len = 5: 571MB
# input_seq_len = 50: 1583MB
# input_seq_len = 100: 2700MB
# input_seq_len = 200: 4957MB
# input_seq_len = 300: 7200MB
# input_seq_len = 400: 9500MB
# input_seq_len = 500: OOM


def iterate_batches(batches_to_make, batch_size):
    for _ in range(batches_to_make):
        batch = []
        for _ in range(batch_size):
            output = [random.randrange(INPUT_SIZE) for _ in range(INPUT_SEQ_LEN)]
            batch.append(output)
        yield np.array(batch, dtype=np.int64)


if __name__ == "__main__":
    net_model = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
    net_out = nn.Linear(in_features=HIDDEN_SIZE, out_features=INPUT_SIZE)
    net_model.cuda()
    net_out.cuda()
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=itertools.chain(net_model.parameters(), net_out.parameters()), lr=0.01)

    for output_batch in tqdm(iterate_batches(BATCHES_TO_ITERATE, BATCH_SIZE), total=BATCHES_TO_ITERATE):
        optimizer.zero_grad()

        sum_loss = None
        rnn_h = None
        next_tokens = [random.randrange(INPUT_SIZE) for _ in range(BATCH_SIZE)]

        for ofs in range(INPUT_SEQ_LEN):
            batch_in = np.zeros(shape=(BATCH_SIZE, 1, INPUT_SIZE), dtype=np.float32)
            for batch_idx, token in enumerate(next_tokens):
                batch_in[batch_idx, 0, token] = 1.0
            batch_in_v = Variable(torch.from_numpy(batch_in)).cuda()

            rnn_out, rnn_h = net_model(batch_in_v, rnn_h)
            out = net_out(rnn_out.squeeze(dim=1))
            next_tokens = torch.multinomial(nn_func.softmax(out), num_samples=1).squeeze().data.cpu().numpy()
            target_idx_v = Variable(torch.from_numpy(output_batch[:, ofs])).cuda()
            loss = objective(out, target_idx_v)
            if sum_loss is None:
                sum_loss = loss
            else:
                sum_loss += loss

            # backpropagate over the head of sequence + encoder steps. This allows to reduce GPU memory footprint
            if ofs == DECODER_HEAD_LEN:
                sum_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                sum_loss = None
                for h in rnn_h:
                    h.detach_()

        # here we backpropagate only through the rest of decoder's steps. This has a downside of short-term dependency
        # of decoded tail on encoder, but we assume that influence of decoder head will be enough.
        # As a positive effect, our GPU memory footprint is predictable now.
        sum_loss.backward()
        optimizer.step()
        pass

    pass
