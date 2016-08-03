-- torch.setheaptracking(true)
require 'hdf5';
require 'rnn';

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'convert_seq/data.hdf5', 'data file')
cmd:option('-testfile', 'convert_seq/data_test.hdf5', 'raw words for test')
cmd:option('-savefile', 'checkpoint_seq/word', 'output file for checkpoints')
cmd:option('-testoutfile', 'seq_test_results_word.hdf5', 'output file for test')
cmd:option('-ltweights', 'embeddings/lstm_LT.h5', 'file containing LT weights')
cmd:option('-gpu', 0, 'whether to use gpu')

-- Hyperparameters
cmd:option('-learning_rate', 1, 'learning rate')
cmd:option('-epochs', 30, 'epochs')
cmd:option('-bsize', 32, 'mini-batch size')
cmd:option('-seqlen', 20, 'seq-len size')
cmd:option('-dhid', 300, 'hidden dimension')
cmd:option('-dwin', 5, 'window size')
cmd:option('-param_init', 0.05, 'initial parameter values')

local data = torch.class('data')
function data:__init(data_file)
   local f = hdf5.open(data_file, 'r')
   self.input = {}
   self.output = {}
   self.lengths = f:read('sent_lens'):all():long()
   self.max_len = f:read('max_len'):all()[1]
   self.nlengths = self.lengths:size(1)
   self.nclasses = f:read('nclasses_chunk'):all():long()[1]
   self.nfeatures = f:read('nfeatures'):all():long()[1]
   for i = 1, self.nlengths do
     local len = self.lengths[i]
     self.input[len] = f:read(tostring(len)):all():double()
     self.output[len] = f:read(tostring(len) .. "_chunks"):all():double()
     if opt.gpu > 0 then
       self.input[len] = self.input[len]:cuda()
       self.output[len] = self.output[len]:cuda()
     end
   end
   f:close()
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

function make_model(train_data, lt_weights) -- batch_size x sentlen tensor input
  local model = nn.Sequential()
  local LT = nn.LookupTable(lt_weights:size(1), lt_weights:size(2))
  LT.weight = lt_weights
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
    -- Assuming data is in format data[sentlen] = { nsent x sentlen tensor, nsent x sentlen tensor }
    for i = 1, train_data.nlengths do
      local sentlen = train_data.lengths[i]
      print(sentlen)
      local nsent = train_data[sentlen][1]:size(1)
      for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
        local batch_idx = (sent_idx - 1) * opt.bsize
        local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
        for col_idx = 1, torch.ceil(sentlen / opt.seqlen) do
          local seq_idx = (col_idx - 1) * opt.seqlen
          local sequence_len = math.min(col_idx * opt.seqlen, sentlen) - seq_idx
          if sequence_len > opt.dwin then
            local train_input_mb = train_data[sentlen][1][{
              { batch_idx + 1, batch_idx + batch_size },
              { seq_idx + 1, seq_idx + sequence_len }}] -- batch_size x senquence_len tensor
            local train_output_mb = train_data[sentlen][2][{
              { batch_idx + 1, batch_idx + batch_size },
              { seq_idx + torch.floor(opt.dwin/2) + 1, seq_idx + sequence_len - torch.floor(opt.dwin/2)}}]
              -- batch_size x (sequence_len - 4)
            train_output_mb = nn.SplitTable(2):forward(train_output_mb) -- (sequence_len - 4) table of batch_size

            criterion:forward(model:forward(train_input_mb), train_output_mb)
            model:zeroGradParameters()
            model:backward(train_input_mb, criterion:backward(model.output, train_output_mb))
            LTgrad:zero()
            model:updateParameters(opt.lambda)
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
  for i = 1, data.nlengths do
    local sentlen = data.lengths[i]
    local nsent = data[sentlen][1]:size(1)
    for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
      local batch_idx = (sent_idx - 1) * opt.bsize
      local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
      for col_idx = 1, torch.ceil(sentlen / opt.seqlen) do
        local seq_idx = (col_idx - 1) * opt.seqlen
        local sequence_len = math.min(col_idx * opt.seqlen, sentlen) - seq_idx
        if sequence_len > opt.dwin then
          local test_input_mb = data[sentlen][1][{
            { batch_idx + 1, batch_idx + batch_size },
            { seq_idx + 1, seq_idx + sequence_len }}] -- batch_size x senquence_len tensor
          local test_output_mb = data[sentlen][2][{
            { batch_idx + 1, batch_idx + batch_size },
            { seq_idx + torch.floor(opt.dwin/2) + 1, seq_idx + sequence_len - torch.floor(opt.dwin/2)}}]
            -- batch_size x (sequence_len - 4)
          test_output_mb = nn.SplitTable(2):forward(test_output_mb)

          nll = nll + criterion:forward(model:forward(test_input_mb), test_output_mb) * batch_size
          total = total + sequence_len * batch_size
        end
      end
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
    if sentlen > opt.dwin then
      table.insert(nlengths, sentlen)
      local test_input = data[sentlen][1] -- nsent x senquence_len tensor
      local test_output = data[sentlen][2][{{},
        { 1 + torch.floor(opt.dwin/2), sentlen - torch.floor(opt.dwin/2)}}] -- batch_size x (sequence_len - 4)
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
  output:write('nlengths', torch.Tensor(nlengths):long())
  output:write('dwin', torch.Tensor{opt.dwin}:long())
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
    local train_data = data.new(opt.datafile)
    local test_data = data.new(opt.testfile)

    -- Create model
    local h = hdf5.open(opt.ltweights, 'r')
		local lt_weights = h:read('weights'):all():double()
    local model, criterion = make_model(train_data, lt_weights)

    -- Train.
    train(train_data, test_data, model, criterion)

    -- Test.
    predict(test_data, model)
end

main()
