-- torch.setheaptracking(true)
require 'hdf5';
require 'rnn';

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'checkpoint_seq/lstm_states.h5', 'data file')
cmd:option('-tagfile', 'convert_seq/data.hdf5', 'tag file for training')
cmd:option('-testfile', 'checkpoint_seq/lstm_states_test.h5', 'raw words for test')
cmd:option('-testtagfile', 'convert_seq/data_test.hdf5', 'tag file for test')
cmd:option('-savefile', 'checkpoint_seq/linear', 'output file for checkpoints')
cmd:option('-testoutfile', 'seq_lin_results.hdf5', 'output file for test')
cmd:option('-gpu', 0, 'whether to use gpu')
cmd:option('-task', 'chunks', 'chunks or pos')

-- Hyperparameters
cmd:option('-learning_rate', 0.01, 'learning rate')
cmd:option('-epochs', 30, 'epochs')
cmd:option('-bsize', 32, 'mini-batch size')
cmd:option('-seqlen', 20, 'seq-len size')
cmd:option('-dhid', 300, 'hidden dimension')
cmd:option('-dwin', 5, 'window dimension')
cmd:option('-param_init', 0.05, 'initial parameter values')

local data = torch.class('data')
function data:__init(data_file, tag_file)
   local f = hdf5.open(data_file, 'r')
   local g = hdf5.open(tag_file, 'r')
   self.input = {}
   self.output = {}
   self.lengths = f:read('sent_lens'):all():long()
   self.nsent = f:read('nsent'):all():long()
   self.nlengths = self.lengths:size(1)
   if opt.task == 'chunks' then
     self.nclasses = g:read('nclasses_chunk'):all():long()[1]
   else
     self.nclasses = g:read('nclasses_pos'):all():long()[1]
   end
   self.state_dim = f:read('state_dim'):all():long()[1]
   -- Load sequencer data from total x 650 state file
   local curr_idx = 1
   local states = f:read('states2'):all()
   for i = 1, self.nlengths do
     local len = self.lengths[i]
     local pad_len = len
     local nsent = self.nsent[i]
     if opt.task == 'chunks' then
       self.output[len] = g:read(tostring(len) .. "_chunks"):all():double()
     else
       self.output[len] = g:read(tostring(len) .. "_pos"):all():double()
     end
     self.input[len] = torch.Tensor(nsent, len, self.state_dim)
     for j = 1, nsent do
       for k = 1, len do
         self.input[len][j][k] = states[curr_idx]
         curr_idx = curr_idx + 1
       end
     end
     if opt.gpu > 0 then
       self.input[len] = self.input[len]:cuda()
       self.output[len] = self.output[len]:cuda()
     end
   end
   f:close()
   g:close()
end

function data.__index(self, idx)
   local input, output
   if type(idx) == "string" then
      return data[idx]
   else
      input = self.input[idx]
      output = self.output[idx]
   end
   return {input, output}
end

function make_model(train_data) -- batch_size x sentlen x state_dim tensor input
  local model = nn.Sequential()
  local lin = nn.Sequential()
  lin:add(nn.SplitTable(1)) -- batch_size table of sentlen x state_dim
  local lin_seq = nn.Sequential()
  lin_seq:add(nn.Linear(train_data.state_dim, opt.dhid)) -- batch_size table of sentlen x hid_dim
  lin_seq:add(nn.Reshape(opt.dhid, 1, true)) -- batch_size table of sentlen x hid_dim x 1
  lin:add(nn.Sequencer(lin_seq))
  lin:add(nn.JoinTable(3)) -- sentlen x hid_dim x batch_size
  model:add(lin)
  model:add(nn.Transpose({2,3})) -- sentlen x batch_size x hid_dim
  model:add(nn.SplitTable(1)) -- (sent_len - 4) table of batch_size x hid_dim
  local seq = nn.Sequential()
  seq:add(nn.HardTanh())
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

function train(train_data, test_data, model, criterion)
  local last_score = 1e9
  local params, gradParams = model:getParameters()
  params:uniform(-opt.param_init, opt.param_init)
  for t = 1, opt.epochs do
    model:training()
    print("Training epoch: " .. t)
    -- Assuming data is in format data[sentlen] = { nsent x sentlen x state_dim tensor, nsent x sentlen tensor }
    for i = 1, train_data.nlengths do
      local sentlen = train_data.lengths[i]
      print(sentlen)
      local d = train_data[sentlen]
      local nsent = d[1]:size(1)
      for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
        local batch_idx = (sent_idx - 1) * opt.bsize
        local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
        local train_input_mb = d[1][{{ batch_idx + 1, batch_idx + batch_size }}] -- batch_size x sequence_len x state_dim tensor
        local train_output_mb = d[2][{{ batch_idx + 1, batch_idx + batch_size }}]
        train_output_mb = nn.SplitTable(2):forward(train_output_mb) -- (sequence_len - 4) table of batch_size

        criterion:forward(model:forward(train_input_mb), train_output_mb)
        model:zeroGradParameters()
        model:backward(train_input_mb, criterion:backward(model.output, train_output_mb))
        model:updateParameters(opt.learning_rate)
      end
    end
    -- Validation error at epoch
    local score = eval(test_data, model, criterion)
    local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, t, score)
    if t == opt.epochs then
      torch.save(savefile, model)
      print('saving checkpoint to ' .. savefile)
    end

    if score > last_score - .01 then
       opt.learning_rate = opt.learning_rate / 2
    end
    last_score = score
    print(t, score, opt.learning_rate)
  end
end

function eval(data, model, criterion)
  model:evaluate()
  local nll = 0
  local total = 0
  for i = 1, data.nlengths do
    local sentlen = data.lengths[i]
    local d = data[sentlen]
    local nsent = d[1]:size(1)
    for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
      local batch_idx = (sent_idx - 1) * opt.bsize
      local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
      local test_input_mb = d[1][{{ batch_idx + 1, batch_idx + batch_size }}]
      local test_output_mb = d[2][{
        { batch_idx + 1, batch_idx + batch_size }}]
      test_output_mb = nn.SplitTable(2):forward(test_output_mb)

      nll = nll + criterion:forward(model:forward(test_input_mb), test_output_mb) * batch_size
      total = total + sentlen * batch_size
    end
    model:forget()
  end
  local valid = math.exp(nll / total)
  return valid
end

function predict(data, model)
  model:evaluate()
  local output = hdf5.open(opt.testoutfile, 'w')
  local accuracy = 0
  local total = 0
  local lengths = {}
  for i = 1, data.nlengths do
    local sentlen = data.lengths[i]
    table.insert(lengths, sentlen)
    local test_input = data[sentlen][1] -- nsent x senquence_len tensor
    local test_output = data[sentlen][2]
    local test_pred = model:forward(test_input)
    local maxidx = {}
    for j = 1, #test_pred do
      _, maxidx[j] = test_pred[j]:max(2)
    end
    maxidx = nn.JoinTable(2):forward(maxidx)
    output:write(tostring(sentlen), maxidx:long())
    output:write(tostring(sentlen) .. '_target', test_output:long())
    accuracy = accuracy + torch.eq(maxidx:long(), test_output:long()):sum()
  end
  output:write('dwin', torch.Tensor{opt.dwin}:long())
  output:write('sent_lens', torch.Tensor(lengths):long())
  accuracy = accuracy / total
  output:write('accuracy', torch.Tensor{accuracy}:double())
  output:close()
  print('Accuracy', accuracy)
end


function main()
   	-- Parse input params
   	opt = cmd:parse(arg)
    if opt.gpu > 0 then
      require 'cudnn';
      require 'cutorch';
      require 'cunn';
    end

    -- Load training data
    local train_data = data.new(opt.datafile, opt.tagfile)
    local test_data = data.new(opt.testfile, opt.testtagfile)

    -- Create model
    local model, criterion = make_model(train_data)

    -- Train.
    train(train_data, test_data, model, criterion)

    -- Test.
    predict(test_data, model)
end

main()
