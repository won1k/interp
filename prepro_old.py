#!/usr/bin/env python

"""Part-Of-Speech Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
import csv

# Global preprocessing variables
start = '<s>'
end = '</s>'
unk = '<unk>'
tag_to_idx = {'B-NP': 1, 'I-NP': 2, 'O': 3}
N = 10000

# Your preprocessing, features construction, and word2vec code.
def get_vocab(words_dict):
    """
    Construct word feature dictionary.
    """
    word_to_idx = {}
    with open(words_dict, 'r') as f:
        f = csv.reader(f, delimiter = ' ', quoting = csv.QUOTE_NONE)
        for row in f:
            word_to_idx[row[0]] = row[1]
    return word_to_idx

def convert_data(state_data, chunk_data, word_to_idx, tag_to_idx, dwin):
    """
    Convert data to windowed word/cap indices.
    """
    with h5py.File(state_data, 'r') as f:
        states = f['states1']
        n = len(states)
        state_dim = len(states[0])
        state_windows = [[] for i in range(n)]

        for idx in range(n):
            print(idx)
            for i in range(-int(dwin/2), dwin/2 + 1):
                if idx + i < 0 or idx + i >= n:
                    state_windows[idx] += [0] * state_dim
                else:
                    state_windows[idx] += list(states[idx + i])

    with h5py.File(chunk_data, 'r') as f:
        chunks = f['chunks']
        tags = []
        for tag in chunks:
            tags.append(tag)

    ntags = len(set(tags))
    return np.array(state_windows, dtype = np.float64), np.array(tags, dtype = np.int32), ntags



FILE_PATHS = {"CoNLL": ("10chunk/states.h5", "10chunk/chunks.h5", "simple_test.txt", "10chunk/words.dict")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help = "Data set", type = str)
    parser.add_argument('window_size', help = "Window size", type = int)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    dwin = int(args.window_size)
    train_states, train_chunks, test, words_dict = FILE_PATHS[dataset]

    # Get word dict
    word_to_idx = get_vocab(words_dict)
    nfeatures = len(word_to_idx)

    # Dataset name
    train_input, train_output, nclasses = convert_data(train_states, train_chunks, word_to_idx, tag_to_idx, dwin)

    #if test:
    #    test_input, test_output = convert_data(test, word_to_idx, tag_to_idx, dwin)
    print("Converted, saving...")
    # Output data
    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['train_input'] = train_input
        f['train_output'] = train_output
        #if test:
        #    f['test_input'] = test_input
        #    f['test_output'] = test_output
        f['nfeatures'] = np.array([nfeatures], dtype = np.int32)
        f['nclasses'] = np.array([nclasses], dtype = np.int32)
        f['dwin'] = np.array([dwin], dtype = np.int32)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
