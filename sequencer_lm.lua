require 'rnn'
require 'hdf5'
require 'nngraph'

cmd = torch.CmdLine()

cmd:option('-data_file','convert_seq/data.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-val_data_file','convert_seq/data_test.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-gpu',1,'which gpu to use. -1 = use CPU')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-savefile', 'checkpoint_seq/lm','filename to autosave the checkpoint to')

cmd:option('-rnn_size', 650, 'size of LSTM internal state')
cmd:option('-word_vec_size', 650, 'dimensionality of word embeddings')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-epochs', 30, 'number of training epoch')
cmd:option('-learning_rate', 1, 'learning rate')
cmd:option('-bsize', 32, 'batch size')
cmd:option('-seqlen', 20, 'sequence length')
cmd:option('-max_grad_norm', 5, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('-dropoutProb', 0.5, 'dropoff param')
cmd:option('-wide', 1, '1 if wide convolution (padded), 0 otherwise')

opt = cmd:parse(arg)


-- Construct the data set.
local data = torch.class("data")
function data:__init(opt, data_file)
   local f = hdf5.open(data_file, 'r')
   self.input = {}
   self.output = {}

   self.lengths = f:read('sent_lens'):all()
   self.max_len = f:read('max_len'):all()[1]
   self.nfeatures = f:read('nfeatures'):all():long()[1]
   self.length = self.lengths:size(1)
   self.dwin = f:read('dwin'):all():long()[1]

   for i = 1, self.length do
     local len = self.lengths[i]
     self.input[len] = f:read(tostring(len)):all():double()
     self.output[len] = f:read(tostring(len) .. "output"):all():double()
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
      input = self.input[idx]:transpose(1,2)
      output = self.output[idx]:transpose(1,2)
   end
   return {input, output}
end


function train(data, valid_data, model, criterion)
   local last_score = 1e9
   local params, grad_params = model:getParameters()
   params:uniform(-opt.param_init, opt.param_init)
   for epoch = 1, opt.epochs do
      model:training()
      print('epoch: ' .. epoch)
      for i = 1, data:size() do 
         local sentlen = data.lengths[i]
         print(sentlen)
         local d = data[sentlen]
         local input, output = d[1], d[2]
         local nsent = input:size(2) -- sentlen x nsent input
         -- If wide convolution, add length for padding
         if opt.wide > 0 then
           sentlen = sentlen + 2 * torch.floor(data.dwin/2)
         end
         for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
           local batch_idx = (sent_idx - 1) * opt.bsize
           local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
           for col_idx = 1, torch.ceil(sentlen / opt.seqlen) do
             local seq_idx = (col_idx - 1) * opt.seqlen
             local sequence_len = math.min(col_idx * opt.seqlen, sentlen) - seq_idx
             local input_mb = input[{
               { seq_idx + 1, seq_idx + sequence_len },
               { batch_idx + 1, batch_idx + batch_size }}] -- sequence_len x batch_size tensor
             local output_mb = output[{
               { seq_idx + 1, seq_idx + sequence_len },
               { batch_idx + 1, batch_idx + batch_size }}]:reshape(sequence_len, batch_size)
             output_mb = nn.SplitTable(1):forward(output_mb) -- sequence_len table of batch_size

             criterion:forward(model:forward(input_mb), output_mb)
             model:zeroGradParameters()
             model:backward(input_mb, criterion:backward(model.output, output_mb))

             -- Grad Norm.
             local grad_norm = grad_params:norm()
             if grad_norm > opt.max_grad_norm then
                grad_params:mul(opt.max_grad_norm / grad_norm)
             end
             params:add(grad_params:mul(-opt.learning_rate))
          end
          model:forget()
        end
      end
      local score = eval(valid_data, model)
      local savefile = string.format('%s_epoch%.2f_%.2f.t7',
                                     opt.savefile, epoch, score)
      --local savefile = string.format('%s_epoch%.2f.t7', opt.savefile, epoch)
      if epoch == opt.epochs then
        torch.save(savefile, model)
        print('saving checkpoint to ' .. savefile)
      end

      if score > last_score - .3 then
         opt.learning_rate = opt.learning_rate / 2
      end
      last_score = score

      print(epoch, score, opt.learning_rate)
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
      local input, output = d[1], d[2]
      local nsent = input:size(2)
      if opt.wide > 0 then
        sentlen = sentlen + 2 * torch.floor(data.dwin/2)
      end
      output = nn.SplitTable(1):forward(output)
      out = model:forward(input)
      nll = nll + criterion:forward(out, output) * nsent
      total = total + sentlen * nsent
      model:forget()
   end
   local valid = math.exp(nll / total)
   return valid
end

function make_model(train_data)
   local model = nn.Sequential()
   model.lookups_zero = {}

   model:add(nn.LookupTable(train_data.nfeatures, opt.word_vec_size))
   model:add(nn.SplitTable(1, 3))

   model:add(nn.Sequencer(nn.FastLSTM(opt.word_vec_size, opt.rnn_size)))
   for j = 2, opt.num_layers do
      model:add(nn.Sequencer(nn.Dropout(opt.dropout_prob)))
      model:add(nn.Sequencer(nn.FastLSTM(opt.rnn_size, opt.rnn_size)))
   end

   model:add(nn.Sequencer(nn.Dropout(opt.dropout_prob)))
   model:add(nn.Sequencer(nn.Linear(opt.rnn_size, train_data.nfeatures)))
   model:add(nn.Sequencer(nn.LogSoftMax()))

   model:remember('both')
   criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

   return model, criterion
end

function main()
    -- parse input params
   opt = cmd:parse(arg)

   if opt.gpu >= 0 then
      print('using CUDA on GPU ' .. opt.gpu .. '...')
      require 'cutorch'
      require 'cunn'
      --cutorch.setDevice(opt.gpu + 1)
   end

   -- Create the data loader class.
   local train_data = data.new(opt, opt.data_file)
   local valid_data = data.new(opt, opt.val_data_file)
   model, criterion = make_model(train_data)

   if opt.gpu >= 0 then
      model:cuda()
      criterion:cuda()
   end
   --torch.save('train_data.t7', train_data)
   --torch.save('valid_data.t7', valid_data)
   train(train_data, valid_data, model, criterion)
end

main()
