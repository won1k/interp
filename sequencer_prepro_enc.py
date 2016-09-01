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
        out.close()

    def text_output(datafile, sentences):
        out = open("sentences_" + datafile, "w")
        idx_to_word = dict([(v, k) for k, v in target_indexer.d.iteritems()])
        for length, sent_list in sentences.iteritems():
            for sentence in sent_list:
                print >>out, ' '.join([idx_to_word[word] for word in sentence])
        out.close()

    def add_padding(sentences, pos_seqs, chunk_seqs, sent_lens, dwin):
        for length in sent_lens:
            for idx, sentence in enumerate(sentences[length]):
                sentences[length][idx] = [target_indexer.convert('PAD')] * (dwin/2) + \
                    sentences[length][idx] + [target_indexer.convert('PAD')] * (dwin/2)
                pos_seqs[length][idx] = [target_indexer.convert_pos('PAD')] * (dwin/2) + \
                    pos_seqs[length][idx] + [target_indexer.convert_pos('PAD')] * (dwin/2)
                chunk_seqs[length][idx] = [target_indexer.convert_chunk('PAD')] * (dwin/2) + \
                    chunk_seqs[length][idx] + [target_indexer.convert_chunk('PAD')] * (dwin/2)
        return sentences, pos_seqs, chunk_seqs

    def convert(datafile, outfile, dwin):
        # Parse and convert data
        with open(datafile, 'r') as f:
            sentences = {}
            sentence = []
            for line in f:
                if not line.strip():
                    length = len(sentence)
                    sentence.append(target_indexer.convert('</s>'))
                    target_indexer.max_len = max(target_indexer.max_len, length)
                    if length in sentences:
                        sentences[length].append(sentence)
                    else:
                        sentences[length] = [sentence]
                    sentence = []
                    continue
                sentence.append(target_indexer.convert(line.strip().split(' ')[0]))
        sent_lens = sentences.keys()

        # Reoutput raw data ordered by length
        text_output(datafile, sentences)

        # Add padding for windowed models
        if dwin > 0:
            sentences = add_padding(sentences, sent_lens, dwin)

        # Output HDF5 for torch
        f = h5py.File(outfile, "w")
        f["sent_lens"] = np.array(sent_lens, dtype=int)
        for length in sent_lens:
            f[str(length)] = np.array(sentences[length], dtype=int)
        f["max_len"] = np.array([target_indexer.max_len], dtype=int)
        f["nfeatures"] = np.array([target_indexer.counter - 1], dtype=int)
        f["dwin"] = np.array([dwin], dtype=int)

    convert(args.trainfile, args.outputfile + ".hdf5", args.dwin)
    target_indexer.lock()
    convert(args.testfile, args.outputfile + "_test" + ".hdf5", args.dwin)
    target_indexer.write(args.outputfile + ".dict")

def main(arguments):
    global args
    parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('trainfile', help="Raw chunking text file", type=str) # train.txt
    parser.add_argument('testfile', help="Raw chunking test text file", type=str) # test.txt
    parser.add_argument('outputfile', help="HDF5 output file", type=str) # convert_seq/data_enc
    args = parser.parse_args(arguments)

    # Do conversion
    get_data(args)

if __name__ == '__main__':
   sys.exit(main(sys.argv[1:]))
