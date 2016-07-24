# interp

# Preprocessing (LSTM / GloVe weights)

## LSTM

1) prepro.py train.txt test.txt 0 convert/train_parsed.txt convert/test_parsed.txt 20 35 convert/datanew
2) model.lua [ >> checkpoint/lm_... ]
3) get_states.lua [ >> checkpoint/lstm_states.h5 ]
4) get_lookuptable.lua [ >> embeddings/lstm_LT.h5 ]

## GloVe

1) prepro_word.py train.txt test.txt 0 convert_word/train_parsed.txt convert_word/test_parsed.txt convert_word/data
2) load_glove.py N convert_word/data.targ.dict [ >> embeddings/gloveN.hdf5 ]

# Pipeline (LSTM States)

1) tempconv.lua [ >> test_results.hdf5 ]
2) postpro_conll.py test.txt convert/train_parsed_chunks.dict test_results.hdf5 test_conll.txt
3) ./conlleval.pl < test_conll.txt

# Pipeline (Word)

# Pipeline (Sequencer)

1) sequencer_prepro.py train.txt test.txt convert_seq/data
2) sequencer_tempconv.lua
