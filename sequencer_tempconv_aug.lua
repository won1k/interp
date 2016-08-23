-- Use word + chunk/pos/feature to predict on LM
-- No windows since features should provide 'context' provided by windows
require 'rnn';
require 'hdf5';

cmd = torch.CmdLine()
-- Files/settings
cmd:option('-datafile','convert_seq/data.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-testfile','convert_seq/data_test.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-savefile', 'checkpoint_seq/lm','filename to autosave the checkpoint to')
cmd:option('-wide', 1, '1 if wide convolution (padded), 0 otherwise')
cmd:option('-gpu', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-feature', 'chunk', 'which feature to use (chunk/pos)')
-- Hyperparameters
cmd:option('-dword', 50, 'dimensionality of word embeddings')
cmd:option('-dfeature', 5, 'dimensionality of feature embeddings')
cmd:option('-dhid', 300, 'dimensionality of hidden layer')
cmd:option('-epochs', 30, 'number of training epoch')
cmd:option('-learning_rate', 1, 'learning rate')
cmd:option('-bsize', 32, 'batch size')
cmd:option('-max_grad_norm', 5, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('-param_init', 0.05, 'initialize parameters at')

opt = cmd:parse(arg)

-- Construct the data set.
local data = torch.class("data")
function data:__init(data_file)
   local f = hdf5.open(data_file, 'r')
   self.input_word = {}
   self.input_feature = {}
   self.output = {}

   self.lengths = f:read('sent_lens'):all()
   self.max_len = f:read('max_len'):all()[1]
   self.nfeatures_word = f:read('nfeatures'):all():long()[1]
   self.nfeatures_feature = f:read('nclasses_' .. opt.feature):all():long()[1]
   self.length = self.lengths:size(1)
   self.dwin = f:read('dwin'):all():long()[1]

   for i = 1, self.length do
     local len = self.lengths[i]
     self.input_word[len] = f:read(tostring(len)):all():double()
     self.input_feature[len] = f:read(tostring(len) .. opt.feature):all():double()
     self.output[len] = f:read(tostring(len) .. "output"):all():double()
     if opt.gpu > 0 then
       self.input_word[len] = self.input_word[len]:cuda()
       self.input_feature[len] = self.input_feature[len]:cuda()
       self.output[len] = self.output[len]:cuda()
     end
   end
   f:close()
end

function data:size()
   return self.length
end

function data.__index(self, idx)
   local input, target
   if type(idx) == "string" then
      return data[idx]
   else
      input_word = self.input_feature[idx]
      input_feature = self.input_feature[idx]
      output = self.output[idx]
   end
   return {input_word, input_feature, output}
end

function make_model(train_data)
  local model = nn.Sequential() -- input: {sentlen x batch_size, sentlen x batch_size}
  local wordLT = nn.LookupTable(train_data.nfeatures_word, opt.dword) -- sentlen x batch_size x dim
  local featureLT = nn.LookupTable(train_data.nfeatures_feature, opt.dfeature)
  model:add(nn.ParallelTable()
    :add(wordLT)
    :add(featureLT)
  ) -- {sentlen x batch_size x dword, sentlen x batch_size x dfeature}
  model:add(nn.JoinTable(3)) -- sentlen x batch_size x (dword + dfeature)
  model:add(nn.SplitTable(1,3)) -- sentlen table of batch_size x (dword + dfeature)
  local nnlm = nn.Sequential()
  nnlm:add(nn.Linear(opt.dword + opt.dfeature, opt.dhid)) -- sentlen table of batch_size x dhid
  nnlm:add(nn.HardTanh())
  nnlm:add(nn.Linear(opt.dhid, train_data.nfeatures_word)) -- sentlen table of batch_size x nfeatures
  nnlm:add(nn.LogSoftMax())
  model:add(nn.Sequencer(nnlm))
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
    print("Training epoch", t)
    for i = 1, train_data.length do
      local sentlen = train_data.lengths[i]
      print(sentlen)
      local d = train_data[sentlen] -- {input_word, input_feature, output}
      local nsent = d[1]:size(1)
      for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
        local batch_idx = (sent_idx - 1) * opt.bsize
        local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
        local input_word_mb = d[1][{{ batch_idx + 1, batch_idx + batch_size }}]:transpose(1,2)
        local input_feature_mb = d[2][{{ batch_idx + 1, batch_idx + batch_size }}]:transpose(1,2)
        local input_mb = {input_word_mb, input_feature_mb}
        local output_mb = d[3][{{ batch_idx + 1, batch_idx + batch_size }}]:reshape(batch_size, sentlen)
        output_mb = nn.SplitTable(2):forward(output_mb)

        criterion:forward(model:forward(input_mb), output_mb)
        model:zeroGradParameters()
        model:backward(input_mb, criterion:backward(model.output, output_mb))
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

    if score > last_score - .001 then
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
  for i = 1, data.length do
    local sentlen = data.lengths[i]
    local d = data[sentlen]
    local nsent = d[1]:size(1)
    for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
      local batch_idx = (sent_idx - 1) * opt.bsize
      local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
      local input_word_mb = d[1][{{ batch_idx + 1, batch_idx + batch_size }}]:transpose(1,2)
      local input_feature_mb = d[2][{{ batch_idx + 1, batch_idx + batch_size }}]:transpose(1,2)
      local output_mb = d[3][{{ batch_idx + 1, batch_idx + batch_size }}]
      output_mb = nn.SplitTable(2):forward(output_mb)

      nll = nll + criterion:forward(model:forward({input_word_mb, input_feature_mb}), output_mb) * batch_size
      total = total + sentlen * batch_size
    end
    model:forget()
  end
  return math.exp(nll / total)
end

function predict(data, model)
  model:evaluate()
  local output = hdf5.open(opt.testoutfile, 'w')
  local accuracy = 0
  local total = 0
  for i = 1, data.length do
    local sentlen = data.lengths[i]
    local d = data[sentlen]
    local nsent = d[1]:size(1)
    local input_word = d[1]:transpose(1,2)
    local input_feature = d[2]:transpose(1,2)
    local output = d[3]
    local test_pred = model:forward({input_word, input_feature})
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
  output:write('dwin', torch.Tensor{opt.dwin}:long())
  accuracy = accuracy / total
  output:write('accuracy', torch.Tensor{accuracy}:double())
  output:close()
  print('Accuracy', accuracy)
end

function main()
   	-- Parse input params
   	opt = cmd:parse(arg)
    if opt.gpu > 0 then
      require 'cutorch';
      require 'cunn';
    end

    -- Load training data
    local train_data = data.new(opt.datafile)
    local test_data = data.new(opt.testfile)

    -- Create model
    local model, criterion = make_model(train_data)

    -- Train.
    train(train_data, test_data, model, criterion)

    -- Test.
    predict(test_data, model)
end

main()
