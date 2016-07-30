require 'rnn'
require 'hdf5'
require 'nngraph'

cmd = torch.CmdLine()

cmd:option('-rnn_size', 650, 'size of LSTM internal state')
cmd:option('-word_vec_size', 650, 'dimensionality of word embeddings')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-epochs', 30, 'number of training epoch')
cmd:option('-learning_rate', 1, '')
cmd:option('-max_grad_norm', 5, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('-dropoutProb', 0.5, 'dropoff param')

cmd:option('-data_file','convert/data.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-val_data_file','convert/datatest.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-savefile', 'checkpoint/lm','filename to autosave the checkpoint to')

opt = cmd:parse(arg)


-- Construct the data set.
local data = torch.class("data")
function data:__init(opt, data_file)
   local f = hdf5.open(data_file, 'r')
   self.input = {}
   self.output = {}

   self.lengths = f:read('sent_lens'):all()[1]
   self.max_len = f:read('max_len'):all()[1]
   self.nclasses = f:read('nclasses'):all():long()[1]
   self.nfeatures = f:read('nfeatures'):all():long()[1]
   self.length = #self.lengths

   for i = 1, self.length do
     local len = self.lengths[i]
     self.input[len] = f:read(tostring(len)):all():double()
     self.output[len] = f:read(tostring(len) .. "_output"):all():double()
     if opt.gpu > 0 then
       self.input[len] = self.input[len]:cuda()
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
      input = self.input[idx]:transpose(1, 2)--:cuda()
      target = nn.SplitTable(2):forward(self.target_output[idx])--:cuda())
      if opt.gpu > 0 then
        input = input:cuda()
        output = output:cuda()
   end
   return {input, target}
end


function train(data, valid_data, model, criterion)
   local last_score = 1e9
   local params, grad_params = model:getParameters()
   params:uniform(-opt.param_init, opt.param_init)
   for epoch = 1, opt.epochs do
      print('epoch: ' .. epoch)
      model:training()
      for i = 1, data:size() do
         local sentlen = data.lengths[i]
         print(sentlen)
         model:zeroGradParameters()
         local d = data[sentlen]
         input, goal = d[1], d[2]
         local out = model:forward(input)
         local loss = criterion:forward(out, goal)
         deriv = criterion:backward(out, goal)
         model:backward(input, deriv)
         -- Grad Norm.
         local grad_norm = grad_params:norm()
         if grad_norm > opt.max_grad_norm then
            grad_params:mul(opt.max_grad_norm / grad_norm)
         end

         params:add(grad_params:mul(-opt.learning_rate))

         if i % 100 == 0 then
            print(i, data:size(),
                  math.exp(loss/ sentlen), opt.learning_rate)
         end
      end
      local score = eval(valid_data, model)
      local savefile = string.format('%s_epoch%.2f_%.2f.t7',
                                     opt.savefile, epoch, score)
      torch.save(savefile, model)
      print('saving checkpoint to ' .. savefile)

      if score > last_score - .3 then
         opt.learning_rate = opt.learning_rate / 2
      end
      last_score = score
   end
end

function eval(data, model)
   -- Validation
   model:evaluate()
   local nll = 0
   local total = 0
   for i = 1, data:size() do
      local sentlen = data.lengths[i]
      local d = data[sentlen]
      local input, goal = d[1], d[2]
      out = model:forward(input)
      nll = nll + criterion:forward(out, goal) * data[sentlen][1]:size(1)
      total = total + sentlen * data[sentlen][1]:size(1)
   end
   local valid = math.exp(nll / total)
   print("Valid", valid)
   return valid
end

function make_model(train_data)
   local model = nn.Sequential()
   model.lookups_zero = {}

   model:add(nn.LookupTable(train_data.nfeatures, opt.word_vec_size))
   model:add(nn.SplitTable(1, 3))

   model:add(nn.Sequencer(nn.FastLSTM(opt.rnn_size, opt.rnn_size)))
   for j = 2, opt.num_layers do
      model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
      model:add(nn.Sequencer(nn.FastLSTM(opt.rnn_size, opt.rnn_size)))
   end

   model:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
   model:add(nn.Sequencer(nn.Linear(opt.rnn_size, train_data.nclasses)))
   model:add(nn.Sequencer(nn.LogSoftMax()))

   model:remember('both')
   criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

   return model, criterion
end

function main()
    -- parse input params
   opt = cmd:parse(arg)

   if opt.gpuid >= 0 then

      print('using CUDA on GPU ' .. opt.gpuid .. '...')
      require 'cutorch'
      require 'cunn'
      require 'cudnn'
      cutorch.setDevice(opt.gpuid + 1)
   end

   -- Create the data loader class.
   local train_data = data.new(opt, opt.data_file)
   local valid_data = data.new(opt, opt.val_data_file)
   model, criterion = make_model(train_data)

   if opt.gpuid >= 0 then
      model:cuda()
      criterion:cuda()
   end
   --torch.save('train_data.t7', train_data)
   --torch.save('valid_data.t7', valid_data)
   train(train_data, valid_data, model, criterion)
end

main()
