require 'rnn'
require 'hdf5'
require 'nngraph'

cmd = torch.CmdLine()

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
cmd:option('-auto', 1, '1 if autoencoder (i.e. target = source), 0 otherwise')

cmd:option('-data_file','convert_seq/data_enc.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-val_data_file','convert_seq/data_enc_test.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-gpu', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-savefile', 'checkpoint_seq/enc','filename to autosave the checkpoint to')

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
   if opt.auto == 1 then
     self.nclasses = f:read('nfeatures'):all():long()[1]
   else
     self.nclasses = f:read('nclasses'):all():long()[1]
   end
   self.length = self.lengths:size(1)
   self.dwin = opt.dwin

   for i = 1, self.length do
     local len = self.lengths[i]
     self.input[len] = f:read(tostring(len)):all():double()
     if opt.auto == 1 then
       self.output[len] = self.input[len]
     else
       self.output[len] = f:read(tostring(len) .. "_target"):all():double()
     end
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

-- Connect functions for encoder-decoder
function forwardConnect(enc, dec)
   for i = 1, #enc.lstmLayers do
      local seqlen = #enc.lstmLayers[i].outputs
      dec.lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, enc.lstmLayers[i].outputs[seqlen])
      dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, enc.lstmLayers[i].cells[seqlen])
   end
end

function backwardConnect(enc, dec)
   for i = 1, #enc.lstmLayers do
      enc.lstmLayers[i].userNextGradCell = nn.rnn.recursiveCopy(enc.lstmLayers[i].userNextGradCell, dec.lstmLayers[i].userGradPrevCell)
      enc.lstmLayers[i].gradPrevOutput = nn.rnn.recursiveCopy(enc.lstmLayers[i].gradPrevOutput, dec.lstmLayers[i].userGradPrevOutput)
   end
end

function storeState(dec)
  for i = 1, #dec.lstmLayers do
    dec.lstmLayers[i].userPrevOutput = dec.lstmLayers[i].output
    dec.lstmLayers[i].userPrevCell = dec.lstmLayers[i].cell
  end
end


function train(data, valid_data, encoder, decoder, criterion)
   local last_score = 1e9
   local encParams, encGradParams = encoder:getParameters()
   local decParams, decGradParams = decoder:getParameters()
   encParams:uniform(-opt.param_init, opt.param_init)
   decParams:uniform(-opt.param_init, opt.param_init)

   for epoch = 1, opt.epochs do
      encoder:training()
      decoder:training()
      print('epoch: ' .. epoch)
      local trainErr = 0
      for i = 1, data:size() do
         local sentlen = data.lengths[i]
         print(sentlen)
         local d = data[sentlen]
         local input, output = d[1], d[2]
         local nsent = input:size(2) -- sentlen x nsent input
         for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
           local batch_idx = (sent_idx - 1) * opt.bsize
           local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
           local input_mb = input[{{1, sentlen}, { batch_idx + 1, batch_idx + batch_size }}] -- sentlen x batch_size tensor
           local output_mb = output[{{}, { batch_idx + 1, batch_idx + batch_size }}]
           output_mb = nn.SplitTable(1):forward(output_mb) -- sentlen table of batch_size

           -- Encoder forward prop
           local encoderOutput = encoder:forward(input_mb) -- sentlen table of batch_size x rnn_size
           -- Decoder forward prop
           forwardConnect(encoder, decoder)
           local decoderInput = { input[{{sentlen + 1}, { batch_idx + 1, batch_idx + batch_size }}] }
           local decoderOutput = { decoder:forward(decoderInput[1])[1] }
           for t = 2, #output_mb do
             local _, nextInput = decoderOutput[t-1]:max(2)
             table.insert(decoderInput, nextInput:reshape(1,batch_size))
             storeState(decoder)
             table.insert(decoderOutput, decoder:forward(decoderInput[t])[1])
           end
           decoderInput = nn.JoinTable(1):forward(decoderInput)
           if opt.gpu > 0 then
             decoderInput = decoderInput:cuda()
           else
             decoderInput = decoderInput:double()
           end
           -- Decoder backward prop
           trainErr = trainErr + criterion:forward(decoderOutput, output_mb)
           decoder:zeroGradParameters()
           decoder:backward(decoderInput, criterion:backward(decoder:forward(decoderInput), output_mb))
           -- Encoder backward prop
           encoder:zeroGradParameters()
           backwardConnect(encoder, decoder)
           local encGrads = {}
           for t = 1, #encoderOutput do
             table.insert(encGrads, encoderOutput[t]:zero())
           end
           encoder:backward(input_mb, encGrads)

           -- Grad norm and update
           local encGradNorm = encGradParams:norm()
           local decGradNorm = decGradParams:norm()
           if encGradNorm > opt.max_grad_norm then
              encGradParams:mul(opt.max_grad_norm / encGradNorm)
           end
           if decGradNorm > opt.max_grad_norm then
              decGradParams:mul(opt.max_grad_norm / decGradNorm)
           end
           encParams:add(encGradParams:mul(-opt.learning_rate))
           decParams:add(decGradParams:mul(-opt.learning_rate))

           encoder:forget()
           decoder:forget()
        end
      end
      print('Training error', trainErr)
      --local score = eval(valid_data, model)
      --local savefile = string.format('%s_epoch%.2f_%.2f.t7',
      --                               opt.savefile, epoch, score)
      local savefile = string.format('%s_epoch%.2f.t7', opt.savefile, epoch)
      --torch.save(savefile, encoder)
      print('saving checkpoint to ' .. savefile)

      --if score > last_score - .3 then
      --   opt.learning_rate = opt.learning_rate / 2
      --end
      --last_score = score
      encoder:forget()
      decoder:forget()
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
   print("Valid", valid)
   return valid
end

function make_model(train_data)
   -- Encoder LSTM
   local encoder = nn.Sequential()
   encoder:add(nn.LookupTable(train_data.nfeatures, opt.word_vec_size)) -- length x bsize x embed
   encoder:add(nn.SplitTable(1,3))
   encoder.lstmLayers = {}
   encoder.lstmLayers[1] = nn.FastLSTM(opt.word_vec_size, opt.rnn_size)
   encoder:add(nn.Sequencer(encoder.lstmLayers[1]))
   for j = 2, opt.num_layers do
      encoder:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
      encoder.lstmLayers[j] = nn.FastLSTM(opt.rnn_size, opt.rnn_size)
      encoder:add(nn.Sequencer(encoder.lstmLayers[j]))
   end
   encoder:remember('both')
   -- Decoder LSTM
   local decoder = nn.Sequential()
   decoder:add(nn.LookupTable(train_data.nfeatures, opt.word_vec_size))
   decoder:add(nn.SplitTable(1,3))
   decoder.lstmLayers = {}
   decoder.lstmLayers[1] = nn.FastLSTM(opt.rnn_size, opt.rnn_size)
   decoder:add(nn.Sequencer(decoder.lstmLayers[1]))
   for j = 2, opt.num_layers do
      decoder:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
      decoder.lstmLayers[j] = nn.FastLSTM(opt.rnn_size, opt.rnn_size)
      decoder:add(nn.Sequencer(decoder.lstmLayers[j]))
   end
   -- Postprocess layers
   decoder:add(nn.Sequencer(nn.Linear(opt.rnn_size, opt.rnn_size)))
   decoder:add(nn.Sequencer(nn.Tanh()))
   decoder:add(nn.Sequencer(nn.Linear(opt.rnn_size, train_data.nclasses)))
   decoder:add(nn.Sequencer(nn.LogSoftMax()))
   decoder:remember('both')
   -- Criterion
   criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
   return encoder, decoder, criterion
end

function main()
    -- parse input params
   opt = cmd:parse(arg)

   if opt.gpu >= 0 then
      print('using CUDA on GPU ' .. opt.gpu .. '...')
      require 'cutorch'
      require 'cunn'
      require 'cudnn'
      --cutorch.setDevice(opt.gpu + 1)
   end

   -- Create the data loader class.
   local train_data = data.new(opt, opt.data_file)
   local valid_data = data.new(opt, opt.val_data_file)
   encoder, decoder, criterion = make_model(train_data)

   if opt.gpu >= 0 then
      encoder:cuda()
      decoder:cuda()
      criterion:cuda()
   end
   --torch.save('train_data.t7', train_data)
   --torch.save('valid_data.t7', valid_data)
   train(train_data, valid_data, encoder, decoder, criterion)
end

main()
