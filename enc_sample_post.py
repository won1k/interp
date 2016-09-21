import h5py
import numpy as np

# Load dict
idx2w = {}
with open('convert_seq/ptb_seq.dict','r') as f:
	f = csv.reader(f)
	for row in f:
		row = row[0].split(' ')
		idx2w[int(row[1])] = row[0]

# Load/translate results
with open('enc_ptb_results_train_words.txt','w') as f:
	train = h5py.File('enc_ptb_results_train.hdf5', 'r')
	for key in train.keys():
		indices = np.transpose(train[key])
		for row in indices:
			sentence = [idx2w[idx] for idx in row]
			sentence = ' '.join(sentence)
			f.write(sentence + '\n')
	train.close()

with open('enc_ptb_results_valid_words.txt','w') as f:
	test = h5py.File('enc_ptb_results_valid.hdf5', 'r')
	for key in test.keys():
		indices = np.transpose(train[key])
		for row in indices:
			sentence = [idx2w[idx] for idx in row]
			sentence = ' '.join(sentence)
			f.write(sentence + '\n')
	test.close()
