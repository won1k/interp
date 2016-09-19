-- Use word + chunk/pos/feature to predict on LM
-- No windows since features should provide 'context' provided by windows
require 'rnn';
require 'hdf5';

cmd = torch.CmdLine()
-- Files/settings
cmd:option('-datafile','convert_seq/data.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-testfile','convert_seq/data_test.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-savefile', 'checkpoint_seq/lm_aug','filename to autosave the checkpoint to')
cmd:option('-testoutfile', 'lm_results.hdf5', 'filename to save results to')
cmd:option('-gpu', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-feature', 'chunk', 'which feature to use (chunk/pos/none)')
-- Hyperparameters
cmd:option('-dword', 50, 'dimensionality of word embeddings')
cmd:option('-dfeature', 5, 'dimensionality of feature embeddings')
cmd:option('-dhid', 300, 'dimensionality of hidden layer')
cmd:option('-epochs', 30, 'number of training epochs (use 0 if until convergence, i.e. low lambda)')
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
   if opt.feature ~= 'none' then
     self.nfeatures_feature = f:read('nclasses_' .. opt.feature):all():long()[1]
   end
   self.length = self.lengths:size(1)
   self.dwin = f:read('dwin'):all():long()[1]
   for i = 1, self.length do
     local len = self.lengths[i]
     self.input_word[len] = f:read(tostring(len)):all():double()
     if opt.feature ~= 'none' then
       self.input_feature[len] = f:read(tostring(len) .. opt.feature):all():double()
     end
     self.output[len] = f:read(tostring(len) .. "output"):all():double()
     if opt.gpu > 0 then
       self.input_word[len] = self.input_word[len]:cuda()
       if opt.feature ~= 'none' then
         self.input_feature[len] = self.input_feature[len]:cuda()
       end
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
     if opt.feature ~= 'none' then
       return {self.input_word[idx], self.input_feature[idx], self.output[idx]}
     else
       return {self.input_word[idx], self.output[idx]}
     end
   end
end

function make_model(train_data)
  local model = nn.Sequential() -- input: {sentlen x batch_size, sentlen x batch_size}
  local wordLT = nn.LookupTable(train_data.nfeatures_word, opt.dword) -- sentlen x batch_size x dim
  local featureLT
  if opt.feature ~= 'none' then
    featureLT = nn.LookupTable(train_data.nfeatures_feature, opt.dfeature)
    model:add(nn.ParallelTable()
      :add(wordLT)
      :add(featureLT)
    ) -- {sentlen x batch_size x dword, sentlen x batch_size x dfeature}
    model:add(nn.JoinTable(3)) -- sentlen x batch_size x (dword + dfeature)
  else
    model:add(wordLT) -- sentlen x batch_size x dword
  end
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

function main()
   	-- Parse input params
   	opt = cmd:parse(arg)
    if opt.gpu > 0 then
      require 'cutorch';
      require 'cunn';
    end
    if opt.feature == 'none' then
      opt.dfeature = 0
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
