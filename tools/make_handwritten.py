import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sudoku_cv.digits import DATA_DIR, normalize_digit

PER_CLASS = 500


def read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    dims = data[3]
    shape = [int.from_bytes(data[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(data, np.uint8, offset=4 + 4 * dims).reshape(shape)


def main():
    mnist_dir = sys.argv[1]
    images = read_idx(os.path.join(mnist_dir, "train-images-idx3-ubyte.gz"))
    labels = read_idx(os.path.join(mnist_dir, "train-labels-idx1-ubyte.gz"))

    rng = np.random.default_rng(0)
    digits, digit_labels = [], []
    for d in range(1, 10):
        for i in rng.choice(np.nonzero(labels == d)[0], PER_CLASS, replace=False):
            mask = (images[i] > 127).astype(np.uint8) * 255
            digits.append(normalize_digit(mask))
            digit_labels.append(d)
    digits = np.uint8(np.round(np.array(digits) * 255))
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(DATA_DIR, "handwritten.npz"), digits=digits,
                        labels=np.array(digit_labels))
    print(f"handwritten: {len(digits)} ({PER_CLASS} per digit)")


if __name__ == "__main__":
    main()
