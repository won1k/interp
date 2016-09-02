require 'rnn'
require 'hdf5'
require 'nngraph'

cmd = torch.CmdLine()

cmd:option('-data_file','convert_seq/data.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-val_data_file','convert_seq/data_test.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-savefile', 'checkpoint_seq/lm','filename to autosave the checkpoint to')
cmd:option('-gpu', 1, 'which gpu to use. 0 = use CPU')
cmd:option('-feature', 'chunk', 'which feature to use (none for no feature)')
cmd:option('-wide', 1, '1 if wide convolution (padded), 0 otherwise')

cmd:option('-dhid', 650, 'size of LSTM internal state')
cmd:option('-dword', 650, 'dimensionality of word embeddings')
cmd:option('-dfeature', 5, 'dimensionality of feature embeddings')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-epochs', 30, 'number of training epoch')
cmd:option('-learning_rate', 1, 'learning rate')
cmd:option('-bsize', 32, 'batch size')
cmd:option('-seqlen', 20, 'sequence length')
cmd:option('-max_grad_norm', 5, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('-dropout_prob', 0.5, 'dropoff param')
cmd:option('-param_init', 0.05, 'initialize parameters at')

opt = cmd:parse(arg)


-- Construct the data set.
local data = torch.class("data")
function data:__init(opt, data_file)
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

function train(data, valid_data, model, criterion)
   local last_score = 1e9
   local params, grad_params = model:getParameters()
   params:uniform(-opt.param_init, opt.param_init)
   for epoch = 1, opt.epochs do
      model:training()
      print('epoch: ' .. epoch)
      local permIdx = torch.randperm(data:size())
      for i = 1, data:size() do
        local sentlen = data.lengths[permIdx[i]]
        io.write("\rSentence length: " .. sentlen)
        io.flush()
        local d = data[sentlen]
        local nsent = d[1]:size(1) -- nsent x sentlen input
        -- If wide convolution, add length for padding
        for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
          local batch_idx = (sent_idx - 1) * opt.bsize
          local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
          local input_mb, output_mb
          if opt.feature ~= 'none' then
            local input_word_mb = d[1][{{ batch_idx + 1, batch_idx + batch_size }}]:transpose(1,2)
            local input_feature_mb = d[2][{{ batch_idx + 1, batch_idx + batch_size }}]:transpose(1,2)
            input_mb = {input_word_mb, input_feature_mb}
            output_mb = d[3][{{ batch_idx + 1, batch_idx + batch_size }}]
          else
            input_mb = d[1][{{ batch_idx + 1, batch_idx + batch_size }}]:transpose(1,2)
            output_mb = d[2][{{ batch_idx + 1, batch_idx + batch_size }}]
          end
          output_mb = nn.SplitTable(2):forward(output_mb)

          criterion:forward(model:forward(input_mb), output_mb)
          model:zeroGradParameters()
          model:backward(input_mb, criterion:backward(model.output, output_mb))

          local grad_norm = grad_params:norm()
          if grad_norm > opt.max_grad_norm then
             grad_params:mul(opt.max_grad_norm / grad_norm)
          end
          params:add(grad_params:mul(-opt.learning_rate))
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
      local paddedlen = sentlen
      if opt.wide > 0 then
        paddedlen = sentlen + 2 * torch.floor(data.dwin/2)
      end
      local d = data[sentlen]
      local nsent = d[1]:size(1)
      local input, output
      if opt.feature ~= 'none' then
        local input_word = d[1]:transpose(1,2)
        local input_feature = d[2]:transpose(1,2)
        input = {input_word, input_feature}
        output = d[3]
      else
        input = d[1]:transpose(1,2)
        output = d[2]
      end
      output = nn.SplitTable(2):forward(output)
      out = model:forward(input)
      nll = nll + criterion:forward(out, output) * nsent
      total = total + paddedlen * nsent
      model:forget()
   end
   local valid = math.exp(nll / total)
   return valid
end

function make_model(train_data)
   local model = nn.Sequential()

   if opt.feature ~= 'none' then
     model:add(nn.ParallelTable()
      :add(nn.LookupTable(train_data.nfeatures_word, opt.dword)) -- batch x sentlen x dword
      :add(nn.LookupTable(train_data.nfeatures_feature, opt.dfeature)) -- batch x sentlen x dfeature
     )
     model:add(nn.JoinTable(3)) -- batch x sentlen x (dword + dfeature)
   else
     model:add(nn.LookupTable(train_data.nfeatures_word, opt.dword))
   end
   model:add(nn.SplitTable(1, 3)) -- batch table of sentlen x (dword + dfeature)

   if opt.feature ~= 'none' then
     model:add(nn.Sequencer(nn.FastLSTM(opt.dword + opt.dfeature, opt.dhid)))
   else
     model:add(nn.Sequencer(nn.FastLSTM(opt.dword, opt.dhid)))
   end

   for j = 2, opt.num_layers do
      model:add(nn.Sequencer(nn.Dropout(opt.dropout_prob)))
      model:add(nn.Sequencer(nn.FastLSTM(opt.dhid, opt.dhid)))
   end

   model:add(nn.Sequencer(nn.Dropout(opt.dropout_prob)))
   model:add(nn.Sequencer(nn.Linear(opt.dhid, train_data.nfeatures_word)))
   model:add(nn.Sequencer(nn.LogSoftMax()))

   model:remember('both')
   criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

   return model, criterion
end

function main()
    -- parse input params
   opt = cmd:parse(arg)
   if opt.gpu > 0 then
     print('using CUDA on GPU ' .. opt.gpu .. '...')
     require 'cutorch'
     require 'cunn'
   end

   -- Create the data loader class.
   local train_data = data.new(opt, opt.data_file)
   local valid_data = data.new(opt, opt.val_data_file)
   model, criterion = make_model(train_data)

   if opt.gpu >= 0 then
      model:cuda()
      criterion:cuda()
   end

   train(train_data, valid_data, model, criterion)
end

main()
