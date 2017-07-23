#!/usr/bin/env python3
import random
import logging
import argparse

TRAIN_FILE = "names"

log = logging.getLogger("train")


def read_data(file_name=TRAIN_FILE, prepend_space=True):
    with open(file_name, "rt", encoding='utf-8') as fd:
        res = map(str.strip, fd.readlines())
        if prepend_space:
            res = map(lambda s: ' ' + s, res)
        return list(res)



if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(name)-14s %(message)s", level=logging.INFO)
    data = read_data()

    log.info("Read %d train samples, first 10: %s", len(data), ', '.join(data[:10]))
    pass
