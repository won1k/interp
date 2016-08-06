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
    sentlens = list(f['sent_lens'])
    dwin = int(f['dwin'][0])
    test_pred = {}
    nsent = {}
    for length in sentlens:
        test_pred[length] = [list(sent) for sent in list(f[str(length)])]
        nsent[length] = len(test_pred[length])
    f.close()

    raw_test = []
    start_idx = 0
    sent_len = 0
    with open(args.rawtestfile, 'r') as f:
        f = csv.reader(f, delimiter = ' ')
        for i, row in enumerate(f):
            if len(row) > 0:
                raw_test.append(row)
                sent_len += 1
            else:
                if sent_len < sentlens[0]:
                    start_idx += sent_len
                sent_len = 0
    raw_test = raw_test[start_idx:]

    chunk_dict = {}
    with open(args.dictfile) as f:
        f = csv.reader(f, delimiter = ' ')
        for row in f:
            chunk_dict[int(row[1])] = row[0]

    output = []
    idx = dwin/2 if args.wide == 0 else 0
    for length in sentlens:
        for sent_idx in range(nsent[length]):
            true_length = length if args.wide > 0 else length - dwin + 1
            for word_idx in range(true_length):
                output.append(raw_test[idx] + [ chunk_dict[test_pred[length][sent_idx][word_idx]] ])
                idx += 1
            output.append([])
            idx += (dwin/2)*2 if args.wide == 0 else 0

    with open(args.outputfile, 'w') as f:
        for row in output:
            f.write(' '.join(row) + '\n')

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rawtestfile', help="Raw chunking test text file", type=str) # sequencer_test.txt
    parser.add_argument('dictfile', help="Chunking tag dictionary file", type=str) # convert_seq/data.chunks.dict
    parser.add_argument('predfile', help="Test chunk tag prediction raw_test file", type=str) # seq_test_results.hdf5
    parser.add_argument('outputfile', help="Text output for conll", type=str) # sequencer_test_conll.txt
    parser.add_argument('wide', help='wide convolution if 1', type=int)
    args = parser.parse_args(arguments)
    postprocess(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
