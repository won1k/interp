import sys
import argparse
import re
import h5py
import numpy as np


def load_data(filename):
   global max_len
   with open(filename, 'r') as f:
      sentences = {}
      tag_seqs = {}
      sentence = []
      tag_seq = []
      length = 0
      for line in f:
         #print(line.strip())
         if not line.strip():
            #print("adding sentence")
            max_len = max(max_len, len(sentence))
            if length in sentences:
               sentences[length].append(sentence)
               tag_seqs[length].append(tag_seq)
            else:
               sentences[length] = [sentence]
               tag_seqs[length] = [tag_seq]
            length = 0
            sentence = []
            tag_seq = []
            continue
         sentence.append(line.strip().split(' ')[:-1])
         tag_seq.append(line.strip().split(' ')[-1])
         length += 1
   return sentences, tag_seqs

def num_features(template):
   num = 0
   with open(template, 'r') as f:
      for line in f:
         if line[0] == '#':
            continue
         elif not line.strip():
            continue
         num += 1
   return num

def add_features(template, sentences):
   #print("execute add_features")
   feat_sentences = {}
   temp = [line for line in open(template, 'r')]
   for i in sentences:
      #print("length", i)
      feat_sentences[i] = [[[feat(pos, sentence, t) for t in temp if feat(pos, sentence, t) is not None]
         for pos in range(len(sentence))] for sentence in sentences[i]]
      '''
      for sentence in sentences[i]:
         # this is a bit slow because you evaluate feat twice, but shouldn't be a huge deal
         feat_sentence = [[feat(pos, sentence, line) for line in open(template, 'r')
            if feat(pos, sentence, line) is not None]
            for pos in range(len(sentence))]
         feat_sentences[i].append(feat_sentence)
      '''
   return feat_sentences

def feat(pos, sentence, feat_template):
   #print(pos, sentence, feat_template)
   def repl(match):
      # gives indices
      idx = [int(i) for i in match.group(1).split(',')]
      if pos+idx[0] < len(sentence) and pos+idx[0] >= 0:
         return sentence[pos+idx[0]][idx[1]]
      else:
         return "PADDING"
   if feat_template[0] == '#':
      return
   elif not feat_template.strip():
      return
   else:
      feat = re.sub('.*?:', '', feat_template)
      return re.sub('%x\[([\-0-9,]*?)\]', repl, feat).strip()

def create_feat_dicts(feat_datasets, tag_datasets):
   feat_dicts = [{} for i in range(num_feats)]
   feat_idx = [0]*num_feats
   tag_dict = {}
   tag_idx = 0
   for i in range(num_feats):
      feat_idx[i] += 1
      feat_dicts[i]["PADDING"] = feat_idx[i]
   tag_idx += 1
   tag_dict["PADDING"] = tag_idx
   for feat_sentences in feat_datasets:
      for length in feat_sentences:
         #print("length", length)
         for feat_sent in feat_sentences[length]:
            #print("feat_sent", feat_sent)
            for feat in feat_sent:
               #print("feat", feat)
               for i in range(len(feat)):
                  #print("feat", i)
                  if feat[i] not in feat_dicts[i]:
                     #print(feat[i])
                     feat_idx[i] += 1
                     feat_dicts[i][feat[i]] = feat_idx[i]
                  #print("feat length", len(feat_dicts[i]))
   for tag_seqs in tag_datasets:
      for length in tag_seqs:
         for tag_seq in tag_seqs[length]:
            #print('seq', tag_seq)
            for tag in tag_seq:
               #print("tag", tag)
               #print("tag_dict", tag_dict)
               if tag not in tag_dict:
                  tag_idx += 1
                  tag_dict[tag] = tag_idx
   return feat_dicts, feat_idx, tag_dict, tag_idx

def data_idx(feat_sentences, feat_dicts):
   for length in feat_sentences:
      #print("idx length", length)
      for feat_sent in feat_sentences[length]:
         #print("idx feat_sent", feat_sent)
         for feat in feat_sent:
            #print("idx feat", feat)
            for i in range(len(feat)):
               feat[i] = feat_dicts[i][feat[i]]
   return feat_sentences

def tag_seq_idx(tag_seqs, tag_dict):
   for length in tag_seqs:
      for i in range(len(tag_seqs[length])):
         tag_seqs[length][i] = [tag_dict[tag] for tag in tag_seqs[length][i]]
   return tag_seqs

def pad_data(idx_sentences):
   data = []
   for length in idx_sentences:
      #print("length", length)
      for sent in idx_sentences[length]:
         #print("sent", sent)
         sent.extend((max_len-length)*[num_feats*[1]])
         #print("data", sent)
         data.append(sent)
   return data

def pad_tag_seqs(tag_seqs):
   gold = []
   for length in tag_seqs:
      for tag_seq in tag_seqs[length]:
         tag_seq.extend((max_len-length)*[1])
         gold.append(tag_seq)
   return gold

def output_dict(dictionary, filename):
   f = open(filename, "w")
   for w in dictionary:
      f.write(str(w) + '\t' + str(dictionary[w]) + '\n')



def main(arguments):
   global args
   parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter)
   parser.add_argument('--template', help="feature template",
                       type=str)
   parser.add_argument('--datafiles', help="data files",
                       type=str, nargs='+')
   parser.add_argument('--outfiles', help="output file",
                       type=str, nargs='+')
   parser.add_argument('--dictfiles', help="dictionary files",
                       type=str, nargs='+')
   args = parser.parse_args(arguments)
   template = args.template
   filenames = args.datafiles
   outfiles = args.outfiles
   dictfiles = args.dictfiles

   assert len(filenames) == len(outfiles), "number of input and output files must be equal"
   for outf in outfiles:
      assert len(outf) >= 5 and outf[-5:] == '.hdf5', "output file names must have hdf5 extension"
   assert len(dictfiles) == 2, "number of dictionary files must be 2"

   global num_feats
   global max_len
   max_len = 0
   num_feats = num_features(template)
   print("number of features", num_feats)

   sentences, tag_seqs, feat_sentences = {}, {}, {}
   feat_sentences = {}
   for f in filenames:
      sentences[f], tag_seqs[f] = load_data(f)
      print("loaded " + f)
      feat_sentences[f] = add_features(template, sentences[f])
      print("added features to " + f)

   feat_dicts, feat_idx, tag_dict, tag_idx = \
      create_feat_dicts([feat_sentences[f] for f in feat_sentences],
         [tag_seqs[f] for f in tag_seqs])
   print("made dictionaries")

   idx_sentences, idx_tags, pad_sentences, pad_tags = {}, {}, {}, {}
   for f in filenames:
      idx_sentences[f] = data_idx(feat_sentences[f], feat_dicts)
      idx_tags[f] = tag_seq_idx(tag_seqs[f], tag_dict)
      pad_sentences[f] = pad_data(idx_sentences[f])
      pad_tags[f] = pad_tag_seqs(idx_tags[f])
      print("indexed data for " + f)

   output_dict(feat_dicts[0], dictfiles[0])
   output_dict(tag_dict, dictfiles[1])
   print("output dictionaries")

   for f, outfile in zip(filenames, outfiles):
      with h5py.File(outfile, 'w') as outf:
         outf['sentences'] = np.array(pad_sentences[f], dtype=int)
         outf['nfeats'] = np.array(feat_idx, dtype=int)
         outf['tags'] = np.array(pad_tags[f], dtype=int)
         outf['ntags'] = np.array([tag_idx], dtype=int)
   print('feature dict sizes', feat_idx)
   print('done.')


if __name__ == '__main__':
   sys.exit(main(sys.argv[1:]))
