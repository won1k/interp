import sys
import argparse
import re
import h5py
import numpy as np

class Indexer:
    def __init__(self):
        self.counter = 1
        self.chunk_counter = 1
        self.pos_counter = 1
        self.d = {}
        self.rev = {}
        self.chunk_d = {}
        self.pos_d = {}
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

    def convert_chunk(self, t):
        if t not in self.chunk_d:
            self.chunk_d[t] = self.chunk_counter
            self.chunk_counter += 1
        return self.chunk_d[t]

    def convert_pos(self, t):
        if t not in self.pos_d:
            self.pos_d[t] = self.pos_counter
            self.pos_counter += 1
        return self.pos_d[t]

    def lock(self):
        self._lock = True

    def write(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k, v
        out.close()

    def write_chunks(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.chunk_d.iteritems()]
        items.sort()
        for v, k in items:
            print >>out, k, v
        out.close()

    def write_pos(self, outfile):
        out = open(outfile, "w")
        items = [(v, k) for k, v in self.pos_d.iteritems()]
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

    def sequencer_template(datafile, sentences, pos_seqs, chunk_seqs):
        out = open("sequencer_" + datafile, "w")
        idx_to_word = dict([(v, k) for k, v in target_indexer.d.iteritems()])
        idx_to_chunk = dict([(v, k) for k, v in target_indexer.chunk_d.iteritems()])
        idx_to_pos = dict([(v, k) for k, v in target_indexer.pos_d.iteritems()])
        for length, sent_list in sentences.iteritems():
            chunk_seq = chunk_seqs[length]
            pos_seq = pos_seqs[length]
            for sent_idx, sentence in enumerate(sent_list):
                for word_idx, word in enumerate(sentence):
                    word = idx_to_word[word]
                    chunk = idx_to_chunk[chunk_seq[sent_idx][word_idx]]
                    pos = idx_to_pos[pos_seq[sent_idx][word_idx]]
                    print >>out, word, pos, chunk
                print >>out, ""

    def convert(datafile, outfile):
        with open(datafile, 'r') as f:
            sentences = {}
            pos_seqs = {}
            chunk_seqs = {}
            sentence = []
            pos_seq = []
            chunk_seq = []
            for line in f:
                if not line.strip():
                    length = len(sentence)
                    target_indexer.max_len = max(target_indexer.max_len, length)
                    if length in sentences:
                        sentences[length].append(sentence)
                        pos_seqs[length].append(pos_seq)
                        chunk_seqs[length].append(chunk_seq)
                    else:
                        sentences[length] = [sentence]
                        pos_seqs[length] = [pos_seq]
                        chunk_seqs[length] = [chunk_seq]
                    sentence = []
                    pos_seq = []
                    chunk_seq = []
                    continue
                sentence.append(target_indexer.convert(line.strip().split(' ')[0]))
                pos_seq.append(target_indexer.convert_pos(line.strip().split(' ')[1]))
                chunk_seq.append(target_indexer.convert_chunk(line.strip().split(' ')[2]))
        f = h5py.File(outfile, "w")
        sent_lens = sentences.keys()
        f["sent_lens"] = np.array(sent_lens, dtype=int)
        for sent_len in sent_lens:
            f[str(sent_len)] = np.array(sentences[sent_len], dtype=int)
            f[str(sent_len) + "_pos"] = np.array(pos_seqs[sent_len], dtype = int)
            f[str(sent_len) + "_chunks"] = np.array(chunk_seqs[sent_len], dtype = int)
        f["max_len"] = np.array([target_indexer.max_len], dtype=int)
        f["nfeatures"] = np.array([target_indexer.counter - 1], dtype=int)
        f["nclasses_pos"] = np.array([target_indexer.pos_counter - 1], dtype=int)
        f["nclasses_chunk"] = np.array([target_indexer.chunk_counter - 1], dtype=int)
        sequencer_template(args.trainfile, sentences, pos_seqs, chunk_seqs)

    convert(args.trainfile, args.outputfile + ".hdf5")
    target_indexer.lock()
    convert(args.testfile, args.outputfile + "_test" + ".hdf5")
    target_indexer.write(args.outputfile + ".dict")
    target_indexer.write_chunks(args.outputfile + ".chunks.dict")
    target_indexer.write_pos(args.outputfile + ".pos.dict")

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
