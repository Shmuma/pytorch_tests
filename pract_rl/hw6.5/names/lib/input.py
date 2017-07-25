import numpy as np

TRAIN_FILE = "names"


def read_data(file_name=TRAIN_FILE, prepend_space=True):
    with open(file_name, "rt", encoding='utf-8') as fd:
        res = map(str.strip, fd.readlines())
        if prepend_space:
            res = map(lambda s: ' ' + s, res)
        return list(res)


class InputEncoder:
    def __init__(self, data):
        self.alphabet = list(sorted(set("".join(data))))
        self.idx = {c: pos for pos, c in enumerate(self.alphabet)}

    def __len__(self):
        return len(self.alphabet)

    def encode(self, s, width=None):
        assert isinstance(s, str)

        if width is None:
            width = len(s)
        res = np.zeros((width, len(self)), dtype=np.float32)
        for sample_idx, c in enumerate(s):
            idx = self.idx[c]
            res[sample_idx][idx] = 1.0

        return res
