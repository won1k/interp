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
    raw_test = []
    with open(args.rawtestfile, 'r') as f:
        f = csv.reader(f, delimiter = ' ')
        for row in f:
            raw_test.append(row)
    chunk_dict = {}
    with open(args.dictfile) as f:
        f = csv.reader(f, delimiter = ' ')
        for row in f:
            chunk_dict[int(row[1])] = row[0]
    f = h5py.File(args.predfile, 'r')
    test_pred = f['output']
    pred_len = len(test_pred)
    dwin = int(f['dwin'][0])
    output = []
    for i in range(dwin/2, dwin/2 + pred_len):
        if len(raw_test[i]) > 0:
            output.append(raw_test[i] + [chunk_dict[test_pred[i - dwin/2]]])
        else:
            output.append([])
    with open(args.outputfile, 'w') as f:
        for row in output:
            f.write(' '.join(row) + '\n')

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rawtestfile', help="Raw chunking test text file") # test.txt
    parser.add_argument('dictfile', help="Chunking tag dictionary file") # convert/train_parsed_chunks.dict
    parser.add_argument('predfile', help="Test chunk tag prediction raw_test file") # test_results.hdf5
    parser.add_argument('outputfile', help="Text output for conll", type=str) # test_conll.txt
    args = parser.parse_args(arguments)
    postprocess(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
