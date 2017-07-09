import glob
import os.path
import unicodedata
import string


ALL_ASCII_LETTERS = string.ascii_letters + " .,;'"


def all_filenames(data_dir="data"):
    return glob.glob(os.path.join(data_dir, "*.txt"))


def read_file(file_path):
    """
    Read data from file
    :param file_path: 
    :return: 
    """
    with open(file_path, "rt", encoding='utf-8') as fd:
        return list(map(str.rstrip, fd.readlines()))


def name_to_ascii(name):
    """
    Convert unicode name to ASCII equivalent, method taken from http://stackoverflow.com/a/518232/2809427
    :param name: 
    :return: normalized name string
    """
    assert isinstance(name, str)
    return ''.join(
        c for c in unicodedata.normalize('NFD', name)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_ASCII_LETTERS
    )


def read_files(files, normalize_names=True):
    """
    Read train data from given files list
    :param files: list of files, one file per class
    :param normalize_names: if True, read names will be normalized to ASCII
    :return: dict with class -> [lines] data 
    """
    assert isinstance(files, list)
    result = {}

    for path in files:
        class_name = os.path.splitext(os.path.basename(path))[0]

        lines = read_file(path)
        if normalize_names:
            lines = list(map(name_to_ascii, lines))
        result[class_name] = lines

    return result


def read_data(normalize_names=True):
    return read_files(all_filenames(), normalize_names=normalize_names)
