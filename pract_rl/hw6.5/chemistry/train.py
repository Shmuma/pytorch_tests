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


INPUT_LIMIT = 50
OUTPUT_LIMIT = 50

SEED = 2345  # obtained from fair dice roll, do not change!
HIDDEN_SIZE = 512

GRAD_CLIP = 1.0
EPOCHES = 10000
# batch size is in tokens, not in sequences
BATCH_SIZE = 10000
TRAINER_RATIO = 0.5

DECODER_HEADER_ENABLED = True
# length of decoder sequence which will be backpropagated to the encoder
DECODER_HEADER_LEN = 100

log = logging.getLogger("train")


def test_model(net_encoder, net_decoder, input_vocab, output_vocab, data, cuda=False):
    total_tokens = 0
    valid_tokens = 0

    batch_bounds = input.split_batches(data, BATCH_SIZE)
    sample_input = []
    sample_output_true = []
    sample_output = []
    for batch_start, batch_end in tqdm(batch_bounds):
        batch = data[batch_start:batch_end]

        input_packed, output_sequences = input.encode_batch(batch, input_vocab, cuda=cuda, volatile=True)
        enc_out, hid = net_encoder(input_packed)
        end_token_idx = output_vocab.token_index[input.END_TOKEN]
        input_token_indices = torch.LongTensor([end_token_idx]).repeat(len(batch), 1)
        max_out_len = max(map(len, output_sequences)) + 1  # extra item for final END_TOKEN
        input_emb = torch.FloatTensor(len(batch), len(output_vocab))

        if cuda:
            input_emb = input_emb.cuda()
            input_token_indices = input_token_indices.cuda()

        dec_hid = None
        dec_attn = net_decoder.initial_attention(batch_size=len(batch), cuda=cuda)
        fill_sample = False
        if not sample_input:
            fill_sample = True
            sample_input = list(batch[0][0])
            sample_output_true = list(batch[0][1])

        for ofs in range(max_out_len):
            input_emb.zero_()
            input_emb.scatter_(1, input_token_indices, 1.0)
            input_emb_v = Variable(input_emb, volatile=True)
            if cuda:
                input_emb_v = input_emb_v.cuda()
            dec_input = torch.cat([input_emb_v, dec_attn, hid], dim=1)

            dec_out, dec_attn, dec_hid = net_decoder(dec_input, dec_hid, enc_out)
            input_token_indices = torch.multinomial(nn_func.softmax(dec_out), num_samples=1).data
            for seq_idx, token in enumerate(input_token_indices):
                cur_seq = output_sequences[seq_idx]
                if ofs > len(cur_seq):
                    continue
                total_tokens += 1
                if ofs == len(cur_seq):
                    valid_tokens += int(token[0] == end_token_idx)
                else:
                    valid_tokens += int(token[0] == output_vocab.token_index[cur_seq[ofs]])
                    if fill_sample and seq_idx == 0:
                        sample_output.append(output_vocab.index_token[token[0]])
    if output_vocab.is_words:
        sample_input = " ".join(sample_input)
        sample_output = " ".join(sample_output)
        sample_output_true = " ".join(sample_output_true)
    else:
        sample_input = "".join(sample_input)
        sample_output = "".join(sample_output)
        sample_output_true = "".join(sample_output_true)
    log.info("Sample, true: %s -> %s", sample_input, sample_output_true)
    log.info("Generated: %s", sample_output)

    return valid_tokens / total_tokens


if __name__ == "__main__":
    random.seed(SEED)
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)-14s %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action='store_true', default=False, help="Enable cuda mode")
    parser.add_argument("--name", required=True, help="Name of directory to save models to")
    parser.add_argument("--tiny", default=False, action='store_true', help="Limit amount of samples to 5000")
    parser.add_argument("-d", "--data", choices=('orig', 'first_last', 'count', 'en_fr'), default='orig',
                        help="Dataset to use, default=orig")
    args = parser.parse_args()

    save_path = os.path.join("saves", args.name)
    if args.tiny:
        save_path += "-tiny"
    os.makedirs(save_path, exist_ok=True)

    # read dataset
    words_data = False
    if args.data == "orig":
        data = input.read_data()
    elif args.data == "en_fr":
        data = input.read_en_fr()
        words_data = True
    else:
        data = input.generate_dataset(args.data)
        print(data[:4])
    random.shuffle(data)
    log.info("Have %d samples", len(data))

    input_vocab  = input.Vocabulary(map(lambda p: p[0], data), words=words_data)
    output_vocab = input.Vocabulary(map(lambda p: p[1], data), extra_items=(input.END_TOKEN, ), words=words_data)

    log.info("Input: %s", input_vocab)
    log.info("Output: %s", output_vocab)

    # split into train/test
    train_data, test_data = input.split_train_test(data, ratio=0.9)
    if args.tiny:
        train_data = train_data[:200]
        test_data = test_data[:50]
    # apply input/output limits
    pred = lambda l: list(filter(lambda p: len(p[0]) <= INPUT_LIMIT and len(p[1]) <= OUTPUT_LIMIT, l))
    train_data = pred(train_data)
    test_data = pred(test_data)
    log.info("Train has %d items, test %d", len(train_data), len(test_data))

    # train
    encoder = model.Encoder(input_size=len(input_vocab), hidden_size=HIDDEN_SIZE)
    decoder = model.AttentionDecoder(input_size=encoder.state_size() + HIDDEN_SIZE + len(output_vocab), hidden_size=HIDDEN_SIZE,
                                     output_size=len(output_vocab), max_encoder_input=INPUT_LIMIT)
    if args.cuda:
        encoder.cuda()
        decoder.cuda()
    optimizer = optim.RMSprop(itertools.chain(encoder.parameters(), decoder.parameters()), lr=0.0001)

    end_token_idx = output_vocab.token_index[input.END_TOKEN]
    epoch_losses = []
    epoch_ratios = []
    epoch_ratios_train = []
    total_tokens = sum(map(lambda t: len(t[0]), train_data))

    for epoch in range(EPOCHES):
        losses = []

        ts = time.time()
        random.shuffle(data)
        batch_bounds = input.split_batches(train_data, BATCH_SIZE, max_sequences=16)
        for batch_start, batch_end in tqdm(batch_bounds):
            batch = data[batch_start:batch_end]
            trainer_mode = random.random() < TRAINER_RATIO
            optimizer.zero_grad()
            input_packed, output_sequences = input.encode_batch(batch, input_vocab, cuda=args.cuda)

            enc_out, hid = encoder(input_packed)

            # input for decoder
            input_token_indices = torch.LongTensor([end_token_idx]).repeat(len(batch), 1)
            max_out_len = max(map(len, output_sequences)) + 1 # extra item for final END_TOKEN
            # expected tokens. We encode it at once and transposed, so our valid indices will be at rows
            # TODO: masking https://discuss.pytorch.org/t/how-can-i-compute-seq2seq-loss-using-mask/861/16
            valid_token_indices = np.full(shape=(max_out_len, len(batch)), fill_value=end_token_idx, dtype=np.int64)
            for idx, out_seq in enumerate(output_sequences):
                valid_token_indices[:len(out_seq), idx] = [output_vocab.token_index[c] for c in out_seq]
            valid_token_indices_v = Variable(torch.from_numpy(valid_token_indices))

            input_emb = torch.FloatTensor(len(batch), len(output_vocab))
            if args.cuda:
                input_emb = input_emb.cuda()
                input_token_indices = input_token_indices.cuda()
                valid_token_indices_v = valid_token_indices_v.cuda()

            # iterate decoder over largest sequence
            batch_loss = None
            # decoder hidden state
            dec_hid = None
            # decoder attention state
            dec_attn = decoder.initial_attention(batch_size=len(batch), cuda=args.cuda)

            for ofs in range(max_out_len):
                # fill input embeddings with one-hot
                input_emb.zero_()
                input_emb.scatter_(1, input_token_indices, 1.0)

                # concatenate encoded state with input
                dec_input = torch.cat([Variable(input_emb), dec_attn, hid], dim=1)
                dec_out, dec_attn, dec_hid = decoder(dec_input, dec_hid, enc_out)
                if trainer_mode:
                    input_token_indices = valid_token_indices_v[ofs].data.unsqueeze(dim=1)
                else:
                    # sample next tokens for decoder's input
                    input_token_indices = torch.multinomial(nn_func.softmax(dec_out), num_samples=1).data
                loss = nn_func.cross_entropy(dec_out, valid_token_indices_v[ofs])
                if batch_loss is None:
                    batch_loss = loss
                else:
                    batch_loss += loss

                # backpropagate over the head of sequence + encoder steps. This allows to reduce GPU memory footprint
                if DECODER_HEADER_ENABLED and ofs == DECODER_HEADER_LEN:
                    batch_loss /= DECODER_HEADER_LEN
                    batch_loss.backward()
                    losses.append(batch_loss.cpu().data.numpy())
                    torch.nn.utils.clip_grad_norm(encoder.parameters(), GRAD_CLIP)
                    torch.nn.utils.clip_grad_norm(decoder.parameters(), GRAD_CLIP)
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_loss = None
                    if isinstance(hid, tuple):
                        for h in hid:
                            h.detach_()
                    else:
                        hid.detach_()
                    break

            # here we backpropagate only through the rest of decoder's steps. This has a downside of short-term dependency
            # of decoded tail on encoder, but we assume that influence of decoder head will be enough.
            # As a positive effect, our GPU memory footprint is predictable now.
            if batch_loss is not None:
                batch_loss /= max_out_len
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm(encoder.parameters(), GRAD_CLIP)
                torch.nn.utils.clip_grad_norm(decoder.parameters(), GRAD_CLIP)
                optimizer.step()
                losses.append(batch_loss.cpu().data.numpy())
        test_ratio = test_model(encoder, decoder, input_vocab, output_vocab, test_data, cuda=args.cuda)
        train_ratio = test_model(encoder, decoder, input_vocab, output_vocab, train_data[:len(test_data)], cuda=args.cuda)
        speed = total_tokens / (time.time() - ts)
        log.info("Epoch %d: mean_loss=%.4f, test_ratio=%.3f%%, train_ratio=%.3f%%, speed=%.3f tokens/s",
                 epoch, np.mean(losses), test_ratio * 100.0, train_ratio * 100.0, speed)
        epoch_ratios.append(test_ratio)
        epoch_ratios_train.append(train_ratio)
        epoch_losses.append(np.mean(losses))
        plots.plot_progress(epoch_losses, epoch_ratios, epoch_ratios_train, os.path.join(save_path, "status.html"))
    pass
