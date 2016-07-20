#!/usr/bin/env python

"""Create the data for the LSTM.
"""

import os
import sys
import argparse
import numpy
import h5py
import itertools


class Indexer:
    def __init__(self):
        self.counter = 1
        self.d = {}
        self.rev = {}
        self._lock = False

    def convert(self, w):
        if w not in self.d:
            if self._lock:
                return self.d["<unk>"]
            self.d[w] = self.counter
            self.rev[self.counter] = w
            self.counter += 1
        return self.d[w]

    def lock(self):
        self._lock = True

    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k, v
        out.close()

def get_data(args):
    target_indexer = Indexer()
    #add special words to indices in the target_indexer
    target_indexer.convert("<s>")
    target_indexer.convert("<unk>")
    target_indexer.convert("</s>")

    def parse_chunking(rawfile, targetfile):
        words = []
        postags = []
        chunktags = []
        for row in rawfile:
            row = row.split()
            if len(row) > 0:
                words.append(row[0])
                postags.append(row[1])
                chunktags.append(row[2])
            else:
                words.append('\n')
                postags.append('END')
                chunktags.append('END')
        with open(targetfile, 'w') as f:
            f.write(' '.join(words))
        with open(targetfile.split('.')[0] + '_pos.txt', 'w') as f:
            f.write(' '.join(postags))
        with open(targetfile.split('.')[0] + '_chunks.txt', 'w') as f:
            f.write(' '.join(chunktags))

    def convert(targetfile, batchsize, seqlength, outfile):
        words = []
        wordschar = []
        targets = []
        for i, targ_orig in enumerate(targetfile):
            targ_orig = targ_orig.replace("<eos>", "")
            targ = targ_orig.strip().split() + ["</s>"]
            target_sent = [target_indexer.convert(w) for w in targ]
            words += target_sent

        targ_output = numpy.array(words[1:] + \
                                      [target_indexer.convert("</s>")])
        words = numpy.array(words)
        print (words.shape, "shape of the word array before preprocessing")
        # Write output.
        f = h5py.File(outfile, "w")
        size = words.shape[0] / (batchsize * seqlength)
        print (size, "number of blocks after conversion")

        original_index = numpy.array([i+1 for i, v in enumerate(words)])

        f["target"] = numpy.zeros((size, batchsize, seqlength), dtype=int)
        f["indices"] = numpy.zeros((size, batchsize, seqlength), dtype=int)
        f["target_output"] = numpy.zeros((size, batchsize, seqlength), dtype=int)
        pos = 0
        for row in range(batchsize):
            for batch in range(size):
                f["target"][batch, row] = words[pos:pos+seqlength]
                f["indices"][batch, row] = original_index[pos:pos+seqlength]
                f["target_output"][batch, row] = targ_output[pos:pos+seqlength]
                pos = pos + seqlength
        f["target_size"] = numpy.array([target_indexer.counter])

        f["words"] = words
        f["set_size"] = words.shape[0]

    if args.parsed == 0:
        parse_chunking(args.rawfile, args.targetfile)
        parse_chunking(args.rawtestfile, args.targetvalfile)
        args.targetfile = open(args.targetfile, 'r')
        args.targetvalfile = open(args.targetvalfile, 'r')
    convert(args.targetfile, args.batchsize, args.seqlength, args.outputfile + ".hdf5")
    target_indexer.lock()
    convert(args.targetvalfile, args.batchsize, args.seqlength, args.outputfile + "test" + ".hdf5")
    target_indexer.write(args.outputfile + ".targ.dict")

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rawfile', help="Raw chunking text file",
                        type=argparse.FileType('r')) # train.txt
    parser.add_argument('rawtestfile', help="Raw chunking test text file",
                        type=argparse.FileType('r')) # test.txt
    parser.add_argument('parsed', help="Whether raw file or already parsed",
                        type=int)
    parser.add_argument('targetfile', help="Target Input file") # train_parsed.txt
    parser.add_argument('targetvalfile', help="Target Input Validation file") # test_parsed.txt

    parser.add_argument('batchsize', help="Batchsize",
                        type=int) # 20
    parser.add_argument('seqlength', help="Sequence length",
                        type=int) # 35
    parser.add_argument('outputfile', help="HDF5 output file",
                        type=str) # convert/data.hdf5
    args = parser.parse_args(arguments)
    if args.parsed > 0:
        args.targetfile = open(args.targetfile, 'r')
        args.targetvalfile = open(args.targetvalfile, 'r')
    get_data(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
