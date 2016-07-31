require 'nn'
require 'rnn'
require 'nngraph'
--require 'cutorch'
--require 'cunn'
require 'hdf5'

cmd = torch.CmdLine()
cmd:option('-rnn_size', 650, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-data_file','convert/data.hdf5','path to data file in hdf5 format')
cmd:option('-checkpoint_file','checkpoint/lm_epoch10.00_41.36.t7','path to model checkpoint file in t7 format')
cmd:option('-output_file','checkpoint/lstm_states.h5','path to output LSTM states in hdf5 format')
cmd:option('-gpu',-1,'which gpu to use. -1 = use CPU')
opt = cmd:parse(arg)

if opt.gpuid >= 0 then
   print('using CUDA on GPU ' .. opt.gpuid .. '...')
   require 'cutorch'
   require 'cunn'
   freeMemory, totalMemory = cutorch.getMemoryUsage(1)
end

-- Construct the data set.
local data = torch.class("data")
function data:__init(opt, data_file)
   local f = hdf5.open(data_file, 'r')
   self.input = {}
   self.output = {}

   self.lengths = f:read('sent_lens'):all()
   self.max_len = f:read('max_len'):all()[1]
   self.nclasses = f:read('nclasses'):all():long()[1]
   self.nfeatures = f:read('nfeatures'):all():long()[1]
   self.length = self.lengths:size(1)

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
      input = self.input[idx]:transpose(1, 2)
      output = self.output[idx]:transpose(1,2)
   end
   return {input, output}
end

local data = data.new(opt.data_file)
model = torch.load(opt.checkpoint_file)

k = 1
Module = nn.Module
all_hidden = {}
count = {}
total_count = 0
for i = 1, data.length do
   local len = data.lengths[i]
   local nsent = data.input[len]:size(1)
   total_count = total_count + len * nsent
end

for i = 1, (2*opt.num_layers) do
   all_hidden[i] = torch.CudaTensor(total_count, opt.rnn_size)
   count[i] = 1
end

function Module:get_states(batch_idx)
   if self.modules then
      for i, module in ipairs(self.modules) do
         if torch.type(module) == "nn.FastLSTM" then
            if module.output ~= nil then
               all_hidden[k][count[k]]:copy(module.output[batch_idx])
               count[k] = count[k] + 1
               k = k + 1
            end
            if module.cell ~= nil then
               all_hidden[k][count[k]]:copy(module.cell[batch_idx])
               count[k] = count[k] + 1
               k = k + 1
            end
         else
            module:get_states()
         end
      end
   end
end

function eval1(data, model)
  model:forget()
  model:evaluate()
  for i = 1, data.length do
     local sentlen = data.lengths[i]
     local d = data[sentlen] -- sent_len x nsent tensors
  	 local input, goal = d[1], d[2]
     local nsent = input:size(2)
     for b = 1, nsent do
       for j = 1, sentlen do
          out = model:forward(input:narrow(1,j,1)) -- 1 x nsent tensor
          k = 1
          model:get_states(b)
       end
     end
     print('output sentence ' .. b .. ' of ' .. nsent)
  end
end

print(all_hidden[1]:size())
eval1(data, model)
local f = hdf5.open(opt.output_file, 'w')
f:write('output1', all_hidden[1]:float())
f:write('states1', all_hidden[2]:float())
f:write('output2', all_hidden[3]:float())
f:write('states2', all_hidden[4]:float())
f:close()