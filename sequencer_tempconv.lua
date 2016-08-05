-- torch.setheaptracking(true)
require 'hdf5';
require 'rnn';

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'checkpoint_seq/lstm_states.h5', 'data file')
cmd:option('-tagfile', 'convert_seq/data.hdf5', 'tag file for training')
cmd:option('-testfile', 'checkpoint_seq/lstm_states_test.h5', 'raw words for test')
cmd:option('-testtagfile', 'convert_seq/data_test.hdf5', 'tag file for test')
cmd:option('-savefile', 'checkpoint_seq/tempconv', 'output file for checkpoints')
cmd:option('-testoutfile', 'seq_test_results.hdf5', 'output file for test')
cmd:option('-gpu', 0, 'whether to use gpu')
cmd:option('-wide', 1, '1 if wide convolution (padded), 0 otherwise')

-- Hyperparameters
cmd:option('-learning_rate', 0.01, 'learning rate')
cmd:option('-epochs', 30, 'epochs')
cmd:option('-bsize', 32, 'mini-batch size')
cmd:option('-seqlen', 20, 'seq-len size')
cmd:option('-dhid', 300, 'hidden dimension')
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
   self.nclasses = f:read('nclasses'):all():long()[1]
   self.state_dim = f:read('state_dim'):all():long()[1]
   self.dwin = g:read('dwin'):all():long()[1]

   -- Load sequencer data from total x 650 state file
   local curr_idx = 1
   local states = f:read('states2'):all()
   for i = 1, self.nlengths do
     local len = self.lengths[i]
     local nsent = self.nsent[i]
     self.input[len] = torch.Tensor(nsent, len, self.state_dim)
     self.output[len] = g:read(tostring(len) .. "_chunks"):all():double()
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
  local temp = nn.Sequential()
  temp:add(nn.SplitTable(1)) -- batch_size table of sentlen x state_dim
  local temp_seq = nn.Sequential()
  temp_seq:add(nn.TemporalConvolution(train_data.state_dim, opt.dhid, train_data.dwin)) -- batch_size table of (sent_len - 4) x hid_dim
  temp_seq:add(nn.Reshape(opt.dhid, 1, true)) -- batch_size table of (sent_len - 4) x hid_dim x 1
  temp:add(nn.Sequencer(temp_seq))
  temp:add(nn.JoinTable(3)) -- (sent_len - 4) x hid_dim x batch_size
  model:add(temp)
  model:add(nn.Transpose({2,3})) -- (sent_len - 4) x batch_size x hid_dim
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
  -- Get params to prevent LT weights update
  local LTweights, LTgrad = model:get(1):getParameters()
  for t = 1, opt.epochs do
    model:training()
    print("Training epoch: " .. t)
    -- Assuming data is in format data[sentlen] = { nsent x sentlen x state_dim tensor, nsent x sentlen tensor }
    for i = 1, train_data.nlengths do
      local sentlen = train_data.lengths[i]
      print(sentlen)
      local nsent = train_data[sentlen][1]:size(1)
      if opt.wide > 0 then
        sentlen = sentlen + 2 * torch.floor(train_data.dwin/2)
      end
      for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
        local batch_idx = (sent_idx - 1) * opt.bsize
        local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
        for col_idx = 1, torch.ceil(sentlen / opt.seqlen) do
          local seq_idx = (col_idx - 1) * opt.seqlen
          local sequence_len = math.min(col_idx * opt.seqlen, sentlen) - seq_idx
          if sequence_len > train_data.dwin then
            local train_input_mb = train_data[sentlen][1][{
              { batch_idx + 1, batch_idx + batch_size },
              { seq_idx + 1, seq_idx + sequence_len }}] -- batch_size x sequence_len x state_dim tensor
            local train_output_mb = train_data[sentlen][2][{
              { batch_idx + 1, batch_idx + batch_size },
              { seq_idx + torch.floor(train_data.dwin/2) + 1, seq_idx + sequence_len - torch.floor(train_data.dwin/2)}}]
              -- batch_size x (sequence_len - 4)
            train_output_mb = nn.SplitTable(2):forward(train_output_mb) -- (sequence_len - 4) table of batch_size

            criterion:forward(model:forward(train_input_mb), train_output_mb)
            model:zeroGradParameters()
            model:backward(train_input_mb, criterion:backward(model.output, train_output_mb))
            LTgrad:zero()
            model:updateParameters(opt.learning_rate)
          end
        end
      end
    end
    -- Validation error at epoch
    local score = eval(test_data, model, criterion)
    local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, t, score)
    torch.save(savefile, model)
    print('saving checkpoint to ' .. savefile)

    if score > last_score - .3 then
       opt.learning_rate = opt.learning_rate / 2
    end
    last_score = score
  end
end

function eval(data, model, criterion)
  model:evaluate()
  local nll = 0
  local total = 0
  local start_idx = data.dwin
  for i = 1, data.nlengths do
    local sentlen = data.lengths[i]
    local nsent = data[sentlen][1]:size(1)
    if opt.wide > 0 then
      sentlen = sentlen + 2 * torch.floor(data.dwin/2)
      start_idx = 0
    end
    for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
      local batch_idx = (sent_idx - 1) * opt.bsize
      local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
      local test_input_mb = data[sentlen][1][{
        { batch_idx + 1, batch_idx + batch_size }}] -- batch_size x senquence_len tensor
      local test_output_mb = data[sentlen][2][{
        { batch_idx + 1, batch_idx + batch_size }}]
            -- batch_size x (sequence_len - 4)
      test_output_mb = nn.SplitTable(2):forward(test_output_mb)

      nll = nll + criterion:forward(model:forward(test_input_mb), test_output_mb) * batch_size
      total = total + sequence_len * batch_size
    end
    model:forget()
  end
  local valid = math.exp(nll / total)
  print('Validation error', valid)
  return valid
end

function predict(data, model)
  model:evaluate()
  local output = hdf5.open(opt.testoutfile, 'w')
  local accuracy = 0
  local total = 0
  local nlengths = {}
  for i = 1, data.nlengths do
    local sentlen = data.lengths[i]
    if sentlen > data.dwin then
      table.insert(nlengths, sentlen)
      local test_input = data[sentlen][1] -- nsent x senquence_len tensor
      local test_output = data[sentlen][2][{{},
        { 1 + torch.floor(data.dwin/2), sentlen - torch.floor(data.dwin/2)}}] -- batch_size x (sequence_len - 4)
      local test_pred = model:forward(test_input)
      local maxidx = {}
      for j = 1, #test_pred do
        _, maxidx[j] = test_pred[j]:max(2)
      end
      maxidx = nn.JoinTable(2):forward(maxidx)
      output:write(tostring(sentlen), maxidx:long())
      output:write(tostring(sentlen) .. '_target', test_output:long())
      accuracy = accuracy + torch.eq(maxidx:long(), test_output:long()):sum()
      total = total + test_output:long():ge(0):sum()
    end
  end
  output:write('dwin', torch.Tensor{data.dwin}:long())
  output:write('nlengths', torch.Tensor(nlengths):long())
  print('Accuracy', accuracy / total)
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
