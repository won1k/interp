import sys
import argparse
import re
import h5py
import numpy as np

class Indexer:
    def __init__(self):
        self.counter = 1
        self.tag_counter = 1
        self.d = {}
        self.rev = {}
        self.tag_d = {}
        self._lock = False
        self.max_len = 0

    def convert(self, w):
        if w not in self.d:
            if self._lock:
                return self.d["<unk>"]
            self.d[w] = self.counter
            self.rev[self.counter] = w
            self.counter += 1
        return self.d[w]

    def convert_tag(self, t):
        if t not in self.tag_d:
            self.tag_d[t] = self.tag_counter
            self.tag_counter += 1
        return self.tag_d[t]

    def lock(self):
        self._lock = True

    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k, v
        out.close()

    def write_tags(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.tag_d.iteritems()]
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

    def convert(datafile, outfile):
        with open(args.trainfile, 'r') as f:
            sentences = {}
            tag_seqs = {}
            sentence = []
            tag_seq = []
            for line in f:
                if not line.strip():
                    length = len(sentence)
                    target_indexer.max_len = max(target_indexer.max_len, length)
                    if length in sentences:
                        sentences[length].append(sentence)
                        tag_seqs[length].append(tag_seq)
                    else:
                        sentences[length] = [sentence]
                        tag_seqs[length] = [tag_seq]
                    sentence = []
                    tag_seq = []
                    continue
                sentence.append(target_indexer.convert(line.strip().split(' ')[0]))
                tag_seq.append(target_indexer.convert_tag(line.strip().split(' ')[-1]))
        f = h5py.File(outfile, "w")
        sent_lens = sentences.keys()
        f["sent_lens"] = np.array(sent_lens, dtype=int)
        for sent_len in sent_lens:
            f[str(sent_len)] = np.array(sentences[sent_len], dtype=int)
            f[str(sent_len) + "_output"] = np.array(tag_seqs[sent_len], dtype = int)
        f["max_len"] = np.array([target_indexer.max_len], dtype=int)
        f["nfeatures"] = np.array([target_indexer.counter - 1], dtype=int)
        f["nclasses"] = np.array([target_indexer.tag_counter - 1], dtype=int)

    convert(args.trainfile, args.outputfile + ".hdf5")
    target_indexer.lock()
    convert(args.testfile, args.outputfile + "_test" + ".hdf5")
    target_indexer.write(args.outputfile + ".dict")
    target_indexer.write_tags(args.outputfile + ".tags.dict")

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trainfile', help="Raw chunking text file", type=str) # train.txt
    parser.add_argument('testfile', help="Raw chunking test text file", type=str) # test.txt
    parser.add_argument('outputfile', help="HDF5 output file", type=str) # convert_seq/data
    args = parser.parse_args(arguments)

    # Do conversion
    get_data(args)

if __name__ == '__main__':
   sys.exit(main(sys.argv[1:]))
