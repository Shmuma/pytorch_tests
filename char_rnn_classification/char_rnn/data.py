import os.path
import glob


def all_filenames(data_dir="data"):
    return glob.glob(os.path.join(data_dir, "*.txt"))
