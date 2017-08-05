import pandas as pd

FILE_NAME = "molecules.tsv.gz"


def read_data(file_name=FILE_NAME):
    df = pd.read_csv(file_name, sep='\t')
    is_str = lambda s: type(s) is str
    df = df[df.molecular_formula.apply(is_str) & df.common_name.apply(is_str)]
    filter_fn = lambda s: s.replace("\n", "")
    return df.molecular_formula, list(map(filter_fn, df.common_name))


class Vocabulary:
    """
    Vocabulary which used to encode and decode sequences, built dynamically from datas
    """
    def __init__(self, data, extra_items=()):
        r = set(list("".join(data)))

        self.tokens = list(sorted(r))
        self.tokens.extend(extra_items)
        self.token_index = {token: idx for idx, token in enumerate(self.tokens)}
        self.index_token = {idx: token for idx, token in enumerate(self.tokens)}

    def __repr__(self):
        return "Vocabulary(tokens_len=%d)" % len(self.tokens)

    def __len__(self):
        return len(self.tokens)
