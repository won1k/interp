#!/usr/bin/env python

"""Perplexity scores given output.
"""

import os
import sys
import argparse
import numpy as np
import h5py
import itertools
import csv

def perplexity(args):
    f = h5py.File(args.predfile, 'r')
    test_pred = f['predictions']
    test_len = len(test_pred)
    dwin = int(f['dwin'][0])

    tag_to_idx = {}
    with open(args.chunkdict, 'r') as chunkdict:
        chunkdict = csv.reader(chunkdict, delimiter = ' ')
        for row in chunkdict:
            tag_to_idx[row[0]] = int(row[1])

    chunktags = args.chunkfile.read().strip().split(' ')

    log_perp = 0
    for idx in range(test_len):
        print(test_pred[idx][tag_to_idx[chunktags[idx + dwin/2]]])
        log_perp += test_pred[idx][tag_to_idx[chunktags[idx + dwin/2]] - 1]
    log_perp = -(log_perp / test_len)
    print('Log perplexity: ' + str(log_perp))
    print('Perplexity: ' + str(np.exp(log_perp)))

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('datafile', help="Data file",
                        type=argparse.FileType('r')) # convert/dataval.hdf5
    parser.add_argument('predfile', help="File containing log-prob. predictions") # simple_test_results.hdf5
    args = parser.parse_args(arguments)

    perplexity(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
