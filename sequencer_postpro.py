#!/usr/bin/env python

"""Postprocess to conll.txt format
"""

import os
import sys
import argparse
import numpy
import h5py
import itertools
import csv

def postprocess(args):
    f = h5py.File(args.predfile, 'r')
    sentlens = f['nlengths']
    dwin = int(f['dwin'][0])
    test_pred = {}
    nsent = {}
    for length in sentlens:
        test_pred[length] = f[str(length)]
        nsent[length] = len(test_pred[length])
    f.close()

    raw_test = []
    start_idx = 0
    sent_len = 0
    with open(args.rawtestfile, 'r') as f:
        f = csv.reader(f, delimiter = ' ')
        for i, row in enumerate(f):
            raw_test.append(row)
            if len(row) > 0:
                sent_len += 1
            else:
                if sent_len <= dwin:
                    start_idx = i
                sent_len = 0
    raw_test = raw_test[start_idx + 1:]

    chunk_dict = {}
    with open(args.dictfile) as f:
        f = csv.reader(f, delimiter = ' ')
        for row in f:
            chunk_dict[int(row[1])] = row[0]


    output = []
    len_idx = 0
    sent_idx = 0
    word_idx = 0
    for row in raw_test:
        length = sentlens[len_idx]
        if len(row) > 0:
            output.append(row + [ chunk_dict[ test_pred[length][sent_idx][word_idx] ] ])
            word_idx += 1
            if word_idx > length - dwin:
                word_idx = 0
                sent_idx += 1
            if sent_idx > nsent[length]:
                sent_idx = 0
                len_idx += 1
        else:
            output.append([])

    with open(args.outputfile, 'w') as f:
        for row in output:
            f.write(' '.join(row) + '\n')

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rawtestfile', help="Raw chunking test text file") # sequencer_test.txt
    parser.add_argument('dictfile', help="Chunking tag dictionary file") # convert_seq/train_parsed_chunks.dict
    parser.add_argument('predfile', help="Test chunk tag prediction raw_test file") # seq_test_results.hdf5
    parser.add_argument('outputfile', help="Text output for conll", type=str) # sequencer_test_conll.txt
    args = parser.parse_args(arguments)
    postprocess(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
