#!/usr/bin/env python3
import logging

from chemistry import input


log = logging.getLogger("train")


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)-14s %(message)s", level=logging.INFO)

    # read dataset
    data = input.read_data()
    log.info("Have %d samples", len(data[0]))

    input_vocab = input.Vocabulary(data[0])
    output_vocab = input.Vocabulary(data[1], extra_items=('END', ))

    log.info("Input: %s", input_vocab)
    log.info("Output: %s", output_vocab)

    # split into train/test
    
    # train
    pass
