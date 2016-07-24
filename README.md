# interp

# Preprocessing (LSTM / GloVe weights)

## LSTM

1) prepro.py train.txt test.txt convert/train_parsed.txt convert/test_parsed.txt 20 35 convert/datanew
2) model.lua [ >> checkpoint/lm_... ]
3) get_states.lua [ >> checkpoint/lstm_states.h5 ]
4) get_lookuptable.lua [ >> embeddings/lstm_LT.h5 ]

## GloVe



# Pipeline (Sequencer)

1) sequencer_prepro.py train.txt test.txt convert_seq/data
2) sequencer_tempconv.lua
