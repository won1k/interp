import h5py
import numpy as np
import argparse
import sys

# Command line args
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('dim', help = 'Glove dimension') # 100
parser.add_argument('dict', help = 'Parsed dict') # convert_word/data.targ.dict
args = parser.parse_args(sys.argv[1:])

# Load glove
dim = args.dim
idx = 1
d = {}
glove = []
with open('glove.6B/glove.6B.' + dim + 'd.txt', 'r') as f:
    for line in f:
        line = line.strip().split(' ')
        d[line[0]] = idx
        glove.append(line[1:])
        idx += 1

# Parse using previous dict
parsed_dict = {}
with open(args.dict, 'r') as f:
    for line in f:
        line = line.strip().split(' ')
        parsed_dict[int(line[1])] = line[0]

# Save into hdf5, dictionary
parsed_glove = []
mean_embedding = list(np.mean(np.array(glove, dtype = np.float64), axis = 0))
for i in range(len(parsed_dict)):
    word = parsed_dict[i + 1]
    try:
        parsed_glove.append(glove[d[word] - 1])
    except:
        parsed_glove.append(mean_embedding)

with open('glove.6B/glove' + dim + '_full.dict', 'w') as f:
    items = [(k, v) for k, v in d.iteritems()]
    items.sort()
    for v, k in items:
        print >>f, k, v

f = h5py.File('glove.6B/glove' + dim + '.hdf5', 'w')
f['weights'] = np.array(parsed_glove, dtype = np.float64)
f.close()
