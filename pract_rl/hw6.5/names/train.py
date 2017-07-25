#!/usr/bin/env python3
import numpy as np
import random
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

from lib import input


HIDDEN_SIZE = 128

log = logging.getLogger("train")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)-14s %(message)s", level=logging.INFO)

    data = input.read_data()
    input_encoder = input.InputEncoder(data)

    log.info("Read %d train samples, first 10: %s", len(data), ', '.join(data[:10]))
    pass
