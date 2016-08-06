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

# Your preprocessing, features construction, and word2vec code.
def get_tags(tagfile):
    """
    Construct chunking tags dictionary.
    """
    tag_to_idx = {}
    with open(tagfile, 'r') as f:
        idx = 1
        tags = f.read().split()
        for tag in tags:
            if tag not in tag_to_idx:
                tag_to_idx[tag] = idx
                idx += 1
    return tag_to_idx

def convert_data(tagfile, tag_to_idx):
    """
    Convert data to tags.
    """
    tag_indices = []
    with open(tagfile, 'r') as f:
        tags = f.read().split()
        for tag in tags:
            try:
                tag_indices.append(tag_to_idx[tag])
            except:
                tag_indices.append(tag_to_idx['O'])
    return np.array(tag_indices, dtype = np.int32)

args = {}

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('tagfile', help = "File containing chunking tags", type = str) #convert/train_parsed_chunks.txt
    parser.add_argument('test', help = 'Whether testing tags', type = int)
    parser.add_argument('dictfile', help = 'Dict to use if testing', type = str) #convert/train_parsed_chunks.dict
    args = parser.parse_args(arguments)
    tagfile = args.tagfile

    # Get word dict
    if args.test > 0:
        tag_to_idx = {}
        with open(args.dictfile, 'r') as f:
            f = csv.reader(f, delimiter = ' ')
            for row in f:
                tag_to_idx[row[0]] = row[1]
    else:
        tag_to_idx = get_tags(tagfile)
        with open(tagfile.split('.')[0] + '.dict','w') as f:
            csvfile = csv.writer(f, delimiter = ' ')
            for key, val in tag_to_idx.iteritems():
                csvfile.writerow([key, val])
    nclasses = len(tag_to_idx)

    # Dataset name
    chunk_tags = convert_data(tagfile, tag_to_idx)

    print("Converted, saving...")
    # Output data
    filename = args.tagfile.split('.')[0] + '.hdf5'
    with h5py.File(filename, "w") as f:
        f['tags'] = chunk_tags
        f['nclasses'] = np.array([nclasses], dtype = np.int32)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
