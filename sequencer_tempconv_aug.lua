


function make_model()
  




  local model = nn.Sequential()
  local LT = nn.LookupTable(lt_weights:size(1), lt_weights:size(2))
  model:add(LT) -- batch_size x sentlen x state_dim
  local temp = nn.Sequential()
  temp:add(nn.SplitTable(1)) -- batch_size table of sentlen x state_dim
  local temp_seq = nn.Sequential()
  temp_seq:add(nn.TemporalConvolution(lt_weights:size(2), opt.dhid, opt.dwin)) -- batch_size table of (sent_len - 4) x hid_dim
  temp_seq:add(nn.Reshape(opt.dhid, 1, true)) -- batch_size table of (sent_len - 4) x hid_dim x 1
  temp:add(nn.Sequencer(temp_seq))
  temp:add(nn.JoinTable(3)) -- (sent_len - 4) x hid_dim x batch_size
  model:add(temp)
  model:add(nn.Transpose({2,3})) -- (sent_len - 4) x batch_size x hid_dim
  model:add(nn.SplitTable(1)) -- (sent_len - 4) table of batch_size x hid_dim
  local seq = nn.Sequential()
  seq:add(nn.HardTanh())
  seq:add(nn.Dropout(opt.dropout_prob))
  seq:add(nn.Linear(opt.dhid, train_data.nclasses))
  seq:add(nn.LogSoftMax())
  model:add(nn.Sequencer(seq))
  model:remember('both')

  local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

  if opt.gpu > 0 then
    model:cuda()
    criterion:cuda()
  end

  return model, criterion
end
