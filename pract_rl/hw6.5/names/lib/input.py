import numpy as np

TRAIN_FILE = "names"


def read_data(file_name=TRAIN_FILE):
    with open(file_name, "rt", encoding='utf-8') as fd:
        return list(map(str.strip, fd.readlines()))


class InputEncoder:
    START_TOKEN = '\0'
    END_TOKEN = '\1'

    def __init__(self, data):
        self.alphabet = list(sorted(set("".join(data) + self.END_TOKEN + self.START_TOKEN)))
        self.idx = {c: pos for pos, c in enumerate(self.alphabet)}

    def __len__(self):
        return len(self.alphabet)

    def encode(self, s, width=None):
        assert isinstance(s, str)

        if width is None:
            width = len(s)
        res = np.zeros((width + 2, len(self)), dtype=np.float32)
        for sample_idx, c in enumerate(self.START_TOKEN + s + self.END_TOKEN):
            idx = self.idx[c]
            res[sample_idx][idx] = 1.0

        return res
