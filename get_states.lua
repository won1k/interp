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
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
opt = cmd:parse(arg)

if opt.gpuid >= 0 then

   print('using CUDA on GPU ' .. opt.gpuid .. '...')
   require 'cutorch'
   require 'cunn'
   freeMemory, totalMemory = cutorch.getMemoryUsage(1)
end

-- Construct the data set.
local data = torch.class("data")
function data:__init(data_file)
   local f = hdf5.open(data_file, 'r')
   self.target  = f:read('target'):all()
   self.target_output = f:read('target_output'):all()
   self.target_size = f:read('target_size'):all()[1]
   self.length = self.target:size(1)
   self.batchlength = self.target:size(2)
   self.seqlength = self.target:size(3)
end

function data.__index(self, idx)
   local input, target
   if type(idx) == "string" then
      return data[idx]
   else
      input = self.target[idx]:transpose(1, 2):float()--:cuda()
      target = nn.SplitTable(2):forward(self.target_output[idx]:float())--:cuda())
   end
   return {input, target}
end

local data = data.new(opt.data_file)
model = torch.load(opt.checkpoint_file)

k = 1
currentbatch = 1
Module = nn.Module
all_hidden = {}
count = {}
for i = 1, (2*opt.num_layers) do
   all_hidden[i] = torch.FloatTensor(data.length * data.batchlength * data.seqlength, opt.rnn_size)-- CudaTensor(data.length * data.batchlength * data.seqlength, opt.rnn_size)
   count[i] = 1
end

function Module:get_states()
   if self.modules then
      for i,module in ipairs(self.modules) do
         if torch.type(module) == "nn.FastLSTM" then
            if module.output ~= nil then
               all_hidden[k][count[k]]:copy(module.output[currentbatch])
               count[k] = count[k] + 1
               k = k + 1
            end
            if module.cell ~= nil then
               all_hidden[k][count[k]]:copy(module.cell[currentbatch])
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
   for b = 1, data.batchlength do
      for i = 1, data.length do
         local d = data[i]
      	local input, goal = d[1], d[2]
         for j = 1, data.seqlength do
            out = model:forward(input:narrow(1,j,1))
            k = 1
            model:get_states()
         end
      end
      print('output batch ' .. b .. ' of ' .. data.batchlength)
      currentbatch = currentbatch + 1
   end
end

print(all_hidden[1]:size())
eval1(data, model)
--ret = nn.JoinTable(2):forward(all_hidden)
---print(ret:size())
local f = hdf5.open(opt.output_file, 'w')
--f:write('states', ret)
f:write('output1', all_hidden[1]:float())
f:write('states1', all_hidden[2]:float())
f:write('output2', all_hidden[3]:float())
f:write('states2', all_hidden[4]:float())
f:close()
