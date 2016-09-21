require 'rnn'
require 'hdf5'
require 'nngraph'

cmd = torch.CmdLine()

cmd:option('-rnn_size', 650, 'size of LSTM internal state')
cmd:option('-word_vec_size', 650, 'dimensionality of word embeddings')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-epochs', 30, 'number of training epoch')
cmd:option('-learning_rate', 0.7, 'learning rate')
cmd:option('-start_annealing', 0.5, 'fraction of epochs at which to start annealing learning rate')
cmd:option('-bsize', 32, 'batch size')
cmd:option('-seqlen', 20, 'sequence length')
cmd:option('-max_grad_norm', 5, 'max l2-norm of concatenation of all gradParam tensors')
cmd:option('-auto', 1, '1 if autoencoder (i.e. target = source), 0 otherwise')
cmd:option('-rev', 0, '1 if reversed output, 0 if normal')
cmd:option('-ptb', 0, '1 if ptb')
cmd:option('-adapt', 'none', 'adaptive gradient method (rms/adagrad/adadelta)')
cmd:option('-weight_cost', 0, 'L2 weight decay')
cmd:option('-smooth', 1e8, 'smoothing params')

cmd:option('-data_file','convert_seq/data_enc.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-val_data_file','convert_seq/data_enc_test.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-gpu', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-param_init', 0.05, 'initialize parameters at')
cmd:option('-savefile', 'checkpoint_seq/enc','filename to autosave the checkpoint to')
cmd:option('-loadfile', '', 'filename to load encoder/decoder from, if any')

opt = cmd:parse(arg)

function adaptiveGradient(params, gradParams, gradDenom, gradPrevDenom, prevGrad, adapt)
  -- L2 weight penalization
  gradParams:add(-opt.weight_cost, params)
  -- Adaptive gradient methods
  if adapt == 'rms' then
    gradDenom:cmul(gradParams, gradParams)
    gradPrevDenom:mul(0.9):add(0.1, gradDenom)
    gradDenom = torch.sqrt(gradPrevDenom):add(opt.smooth)
  elseif adapt == 'adagrad' then
    gradDenom:cmul(gradParams, gradParams)
    gradPrevDenom:cmul(gradPrevDenom, gradPrevDenom):add(gradDenom):sqrt()
    gradDenom = gradPrevDenom:clone()
  elseif adapt == 'adadelta' then
    gradDenom:cmul(gradPrevDenom, gradPrevDenom):mul(0.9):addcmul(0.1, gradParams, gradParams):add(opt.smooth):sqrt()
    gradPrevDenom:cmul(prevGrad, prevGrad):add(opt.smooth):sqrt()
    gradDenom:cdiv(gradDenom, gradPrevDenom)
    gradPrevDenom = gradDenom:clone()
  end
  return gradParams, gradDenom, gradPrevDenom
end

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
      --dec.lstmLayers[i].userPrevOutput = enc.lstmLayers[i].outputs[seqlen]:clone()
      --dec.lstmLayers[i].userPrevCell = enc.lstmLayers[i].cells[seqlen]:clone()
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
    dec.lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, dec.lstmLayers[i].output)
    dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, dec.lstmLayers[i].cell)
  end
end


function train(data, valid_data, encoder, decoder, criterion)
   local last_score = 1e9
   local encParams, encGradParams = encoder:getParameters()
   local decParams, decGradParams = decoder:getParameters()
   encParams:uniform(-opt.param_init, opt.param_init)
   decParams:uniform(-opt.param_init, opt.param_init)
   local decGradDenom = torch.ones(decGradParams:size())
   local decGradPrevDenom = torch.zeros(decGradParams:size())
   local decPrevGrad = torch.zeros(decGradParams:size())
   local encGradDenom = torch.ones(encGradParams:size())
   local encGradPrevDenom = torch.zeros(encGradParams:size())
   local encPrevGrad = torch.zeros(encGradParams:size())
   if opt.gpu > 0 then
      decGradDenom = decGradDenom:cuda()
      decGradPrevDenom = decGradPrevDenom:cuda()
      decPrevGrad = decPrevGrad:cuda()
      encGradDenom = encGradDenom:cuda()
      encGradPrevDenom = encGradPrevDenom:cuda()
      encPrevGrad = encPrevGrad:cuda()
   end

   for epoch = 1, opt.epochs do
      print('epoch: ' .. epoch)
      encoder:training()
      decoder:training()
      local trainErr = 0
      local total = 0
      for i = 1, data:size() do
         local sentlen = data.lengths[i]
         print("Sentence length: ", sentlen)
         local d = data[sentlen]
         if opt.ptb > 0 then
          sentlen = sentlen - 1
         end
         local input, output = d[1], d[2]
         local nsent = input:size(2) -- sentlen x nsent input
         --if opt.wide > 0 then
         --  sentlen = sentlen + 2 * torch.floor(data.dwin/2)
         --end
         for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
           local batch_idx = (sent_idx - 1) * opt.bsize
           local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
           local input_mb = input[{{1, sentlen}, { batch_idx + 1, batch_idx + batch_size }}] -- sentlen x batch_size tensor
           local output_mb = output[{{}, { batch_idx + 1, batch_idx + batch_size }}]
           local revInput
           if opt.rev > 0 then
              revInput = {}
              for t = 1, sentlen do
                table.insert(revInput, input_mb[{{sentlen - t + 1},{}}])
              end
              input_mb = nn.JoinTable(1):forward(revInput)
              if opt.gpu > 0 then
                input_mb = input_mb:cuda()
              end
           end

           -- Encoder forward prop
           local encoderOutput = encoder:forward(input_mb) -- sentlen table of batch_size x rnn_size

           -- Decoder forward prop
           forwardConnect(encoder, decoder)
           local decoderInput = torch.cat(input[{{sentlen + 1}, {batch_idx + 1, batch_idx + batch_size}}],
                output_mb[{{1, sentlen}, {}}], 1)
           if opt.gpu > 0 then
             decoderInput = decoderInput:cuda()
           else
             decoderInput = decoderInput:double()
           end
           decoderOutput = decoder:forward(decoderInput)

           -- Decoder backward prop
           output_mb = nn.SplitTable(1):forward(output_mb)
           trainErr = trainErr + criterion:forward(decoderOutput, output_mb) * batch_size
           total = total + sentlen * batch_size
           decoder:zeroGradParameters()
           decoder:backward(decoderInput, criterion:backward(decoderOutput, output_mb))

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
           --decGradParams, decGradDenom, decGradPrevDenom = adaptiveGradient(
           --    decParams, decGradParams, decGradDenom, decGradPrevDenom, decPrevGrad, opt.adapt)
           --encGradParams, encGradDenom, encGradPrevDenom = adaptiveGradient(
           --    encParams, encGradParams, encGradDenom, encGradPrevDenom, encPrevGrad, opt.adapt)
           -- Parameter update
           --decParams:addcdiv(-opt.learning_rate, decGradParams, decGradDenom)
           --decPrevGrad:mul(0.9):addcdiv(0.1, decGradParams, decGradDenom)
           --encParams:addcdiv(-opt.learning_rate, encGradParams, encGradDenom)
           --encPrevGrad:mul(0.9):addcdiv(0.1, encGradParams, encGradDenom)
           encParams:add(encGradParams:mul(-opt.learning_rate))
           decParams:add(decGradParams:mul(-opt.learning_rate))
           encoder:forget()
           decoder:forget()
        end
      end
      print('Training error', trainErr / total)
      local score = eval(valid_data, encoder, decoder)
      local savefile = string.format('%s_epoch%.2f_%.2f',
                                    opt.savefile, epoch, score)
      if epoch % (opt.epochs/2) == 0 then
        torch.save(savefile .. 'encoder.t7', encoder)
        torch.save(savefile .. 'decoder.t7', decoder)
        print('saving checkpoint to ' .. savefile)
      end

      if score > last_score - .3 and epoch > opt.start_annealing * opt.epochs then
         opt.learning_rate = opt.learning_rate / 2
      end
      last_score = score
      encoder:forget()
      decoder:forget()
      print(epoch, score, opt.learning_rate)
   end
end

function eval(data, encoder, decoder)
   -- Validation
   encoder:evaluate()
   decoder:evaluate()
   local nll = 0
   local total = 0
   local accuracy = 0
   for i = 1, data:size() do
      local sentlen = data.lengths[i]
      local d = data[sentlen]
      local input, output = d[1], d[2]
      if opt.ptb > 0 then
        sentlen = sentlen - 1
      end
      local nsent = input:size(2)
      local revOutput
      if opt.rev > 0 then
        revInput = {}
        for t = 1, sentlen do
          table.insert(revInput, input[{{sentlen - t + 1},{}}])
        end
        input_mb = nn.JoinTable(1):forward(revInput)
        if opt.gpu > 0 then
          input_mb = input_mb:cuda()
        end
      end

      -- Encoder forward prop
      local encoderOutput = encoder:forward(input[{{1, sentlen}}]) -- sentlen table of batch_size x rnn_size
      -- Decoder forward prop
      forwardConnect(encoder, decoder)
      local decoderInput = { input[{{sentlen + 1}}] }
      decoder:remember()
      local decoderOutput = { decoder:forward(decoderInput[1])[1]:clone() }
      for t = 2, sentlen + 1 do
        local _, nextInput = decoderOutput[t-1]:max(2)
        table.insert(decoderInput, nextInput:reshape(1,nsent):clone())
        table.insert(decoderOutput, decoder:forward(decoderInput[t])[1]:clone())
      end
      output = nn.SplitTable(1):forward(output)
      nll = nll + criterion:forward(decoderOutput, output) * nsent
      total = total + sentlen * nsent

      -- Accuracy
      local _, nextInput = decoderOutput[#output]:max(2)
      for t = 1, #output do
        local _, prediction = decoderOutput[t]:max(2)
        prediction = prediction:reshape(nsent)
        for i = 1, nsent do
          if prediction[i] == output[t][i] then
            accuracy = accuracy + 1
          end
        end
      end

      encoder:forget()
      decoder:forget()
   end
   local valid = math.exp(nll / total)
   print("Test error", valid)
   print("Test accuracy", accuracy / total)
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
      --encoder:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
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
      --decoder:add(nn.Sequencer(nn.Dropout(opt.dropoutProb)))
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

   if opt.gpu > 0 then
      print('using CUDA on GPU ' .. opt.gpu .. '...')
      require 'cutorch'
      require 'cunn'
      --require 'cudnn'
      cutorch.setDevice(opt.gpu)
   end

   -- Create the data loader class.
   local train_data = data.new(opt, opt.data_file)
   local valid_data = data.new(opt, opt.val_data_file)

   -- Load models
   local encoder, decoder, criterion = make_model(train_data)
   if opt.loadfile:len() > 0 then
    print("Loading old models...")
      encoder = torch.load(opt.loadfile .. 'encoder.t7')
      decoder = torch.load(opt.loadfile .. 'decoder.t7')
   end

   if opt.gpu > 0 then
      encoder:cuda()
      decoder:cuda()
      criterion:cuda()
   end
   --torch.save('train_data.t7', train_data)
   --torch.save('valid_data.t7', valid_data)
   train(train_data, valid_data, encoder, decoder, criterion)
end

main()
