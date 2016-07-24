-- torch.setheaptracking(true)
require 'hdf5';
require 'rnn';

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'convert_seq/data.hdf5', 'data file')
cmd:option('-testfile', 'convert_seq/data_test.hdf5', 'raw words for test')
cmd:option('-savefile', 'checkpoint_seq/word', 'output file for checkpoints')
cmd:option('-testoutfile', 'seq_test_results.hdf5', 'output file for test')
cmd:option('-ltweights', 'checkpoint/lstm_LT.h5', 'file containing LT weights')
cmd:option('-gpu', 0, 'whether to use gpu')

-- Hyperparameters
cmd:option('-lambda', 0.01, 'learning rate')
cmd:option('-epochs', 30, 'epochs')
cmd:option('-bsize', 32, 'mini-batch size')
cmd:option('-dhid', 300, 'hidden dimension')
cmd:option('-dwin', 5, 'window size')

local data = torch.class('data')
function data:__init(data_file)
   local f = hdf5.open(data_file, 'r')
   self.input = {}
   self.output = {}
   self.lengths = f:read('sent_lens'):all():long()
   self.max_len = f:read('max_len'):all()[1]
   self.nlengths = self.lengths:size(1)
   self.nclasses = f:read('nclasses'):all():long()[1]
   self.nfeatures = f:read('nfeatures'):all():long()[1]
   for i = 1, self.nlengths do
     local len = self.lengths[i]
     self.input[len] = f:read(tostring(len)):all():double()
     self.output[len] = f:read(tostring(len) .. "_output"):all():double()
   end
   if opt.gpu > 0 then
     self.input:cuda()
     self.output:cuda()
   end
   f:close()
end

function data.__index(self, idx)
   local input, output
   if type(idx) == "string" then
      return data[idx]
   else
      input = self.input[idx]--:transpose(1,2):float()
      output = self.output[idx]
      --nn.SplitTable(2):forward(self.output[idx]:float()) -- sent_len table of batch_size
   end
   return {input, output}
end

function make_model(train_data, lt_weights)
  local model = nn.Sequential()
  local LT = nn.LookupTable(lt_weights:size(1), lt_weights:size(2))
  LT.weight = lt_weights
  --model:add(LT)
  --model:add(nn.TemporalConvolution(lt_weights:size(2), opt.dhid, opt.dwin))
  model:add(nn.Transpose({1,2})) -- sent_len x batch_size
  local seq = nn.Sequential()
  seq:add(LT) -- batch_size x state_dim
  seq:add(nn.Linear(lt_weights:size(2), opt.dhid)) -- batch_size x hid_dim
  seq:add(nn.HardTanh())
  seq:add(nn.Linear(opt.dhid, train_data.nclasses)) -- batch_size x nclasses
  seq:add(nn.LogSoftMax())
  local r = nn.Sequencer(seq)
  model:add(r)
  model:remember('both')

  --local LT = nn.LookupTable(lt_weights:size(1), lt_weights:size(2))
  --LT.weight = lt_weights
  --local seq = nn.Sequential() -- PROBLEM: HOW TO DO SEQUENCER WITH TEMP CONV
  --seq:add(LT) -- batch_size x state_dim
  --seq:add(nn.TemporalConvolution(lt_weights:size(2), opt.dhid, opt.dwin))
  --seq:add(nn.HardTanh())
  --seq:add(nn.Linear(opt.dhid, train_data.nclasses))
  --seq:add(nn.LogSoftMax())
  --local net = nn.Sequencer(seq)
  --net:remember('both')

  local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

  if opt.gpu > 0 then
    model:cuda()
    criterion:cuda()
  end

  return model, criterion
end

function train(train_data, test_data, model, criterion)
  model:training()
  -- Get params to prevent LT weights update
  local LTweights, LTgrad = model:get(2):get(1):get(1):get(1):getParameters()
  for t = 1, opt.epochs do
    print("Training epoch: " .. t)
    -- Assuming data is in format data[sentlen] = { nsent x sentlen tensor, nsent x 1 tensor }
    for i = 1, train_data.nlengths do
      local sentlen = train_data.lengths[i]
      if sentlen > opt.dwin then
        print(sentlen)
        local len_data = train_data[sentlen] -- batch_size x sent_len, sentlen table of batch_size
        local train_input, train_output = len_data[1], len_data[2]
        local nsent = train_input:size(1)
        for i = 1, torch.ceil(nsent / opt.bsize) do
          local start_idx = (i - 1) * opt.bsize
          local batch_size = math.min(i * opt.bsize, nsent) - start_idx -- batch_size x sentlen tensor
          local train_input_mb = train_input[{{ start_idx + 1, start_idx + batch_size }}] -- batch_size x sentlen
          local train_output_mb = train_output[{{ start_idx + 1, start_idx + batch_size }}]:transpose(1,2)
          --train_output_mb = train_output_mb[{{}, { torch.floor(opt.dwin/2) + 1, sentlen - torch.floor(opt.dwin/2) }}]:transpose(1,2)
          criterion:forward(model:forward(train_input_mb), train_output_mb)
          model:zeroGradParameters()
          model:backward(train_input_mb, criterion:backward(model.output, train_output_mb))
          LTgrad:zero()
          model:updateParameters(opt.lambda)
        end
      end
    end
    -- Validation error at epoch
    local score = eval(test_data, model, criterion)
    local savefile = string.format('%s_epoch%.2f_%.2f.t7',
                                   opt.savefile, t, score)
    torch.save(savefile, model)
    print('saving checkpoint to ' .. savefile)
  end
end

function eval(data, model, criterion)
  model:evaluate()
  local nll = 0
  local total = 0
  for i = 1, data.nlengths do
    local sentlen = data.lengths[i]
    if sentlen > opt.dwin then
      local len_data = data[sentlen]
      local test_input, test_output = len_data[1], len_data[2]:transpose(1,2)
      --test_output = test_output[{{}, { torch.floor(opt.dwin/2) + 1,
      --                                   sentlen - torch.floor(opt.dwin/2) }}]:transpose(1,2)
      nll = nll + criterion:forward(model:forward(test_input), test_output)
    end
  end
  print('Validation error', nll)
  return nll
end

function predict(data, model)
  model:evaluate()
  local output = hdf5.open(opt.testoutfile, 'w')
  for i = 1, data.nlengths do
    local sentlen = data.lengths[i]
    if sentlen > opt.dwin then
      local len_data = data[sentlen]
      local test_input = len_data[1]
      local test_pred = model:forward(test_input)
      local maxval, maxidx = test_pred:max(2)
      maxidx = maxidx:squeeze()
      output:write(tostring(sentlen), maxidx)
      output:write(tostring(sentlen) .. '_pred', test_pred)
    end
  end
end


function main()
   	-- Parse input params
   	opt = cmd:parse(arg)

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
