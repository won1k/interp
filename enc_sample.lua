require 'rnn'
require 'hdf5'

cmd = torch.CmdLine()
cmd:option('-auto', 1, '1 if autoencoder (i.e. target = source), 0 otherwise')

cmd:option('-data_file','convert_seq/ptb_seq.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-val_data_file','convert_seq/ptb_seq_test.hdf5','data directory. Should contain data.hdf5 with input data')
cmd:option('-gpu', 1, 'which gpu to use. -1 = use CPU')
cmd:option('-savefile', 'enc_samples.hdf5','filename to save samples to')
cmd:option('-loadfile', 'checkpoint_seq/enc_ptb_epoch30.00_33.87', 'filename to load encoder from')

opt = cmd:parse(arg)

function forwardConnect(enc, dec)
   for i = 1, #enc.lstmLayers do
      local seqlen = #enc.lstmLayers[i].outputs
      dec.lstmLayers[i].userPrevOutput = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevOutput, enc.lstmLayers[i].outputs[seqlen])
      dec.lstmLayers[i].userPrevCell = nn.rnn.recursiveCopy(dec.lstmLayers[i].userPrevCell, enc.lstmLayers[i].cells[seqlen])
   end
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
	self.nclasses = f:read('nfeatures'):all():long()[1]
	self.length = self.lengths:size(1)
	self.dwin = opt.dwin
	for i = 1, self.length do
		local len = self.lengths[i]
		self.input[len] = f:read(tostring(len)):all():double()
		self.output[len] = self.input[len]
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

function encodeDecode(data, encoder, decoder, file_name)
	print("Saving to " .. file_name)
	local f = hdf5.open(file_name,'w')
	for i = 1, data:size() do
		local sentlen = data.lengths[i]
		print("Sentence length: ", sentlen)
		local d = data[sentlen]
		local input, output = d[1], d[2]
        local nsent = input:size(2)
        -- Encoder forward
        local encoderOutput = encoder:forward(input[{{1, sentlen - 1}}])
        -- Decoder forward
		forwardConnect(encoder, decoder)
		local decoderInput = { input[{{sentlen}}] }
		decoder:remember()
		local decoderOutput = { decoder:forward(decoderInput[1])[1]:clone() }
		local predictions = {}
		for t = 2, sentlen do
			local _, nextInput = decoderOutput[t-1]:max(2)
			table.insert(decoderInput, nextInput:reshape(1,nsent):clone())
			table.insert(predictions, nextInput:reshape(1,nsent):clone())
			table.insert(decoderOutput, decoder:forward(decoderInput[t])[1]:clone())
		end
		-- Predictions
		local _, lastPred = decoderOutput[sentlen]:max(2)
		table.insert(predictions, lastPred:reshape(1,nsent):clone())
		predictions = nn.JoinTable(1):forward(predictions)
		encoder:forget()
		decoder:forget()
		f:write(tostring(sentlen), predictions)
	end
	f:close()
end

function main()
	-- Check if GPU
	if opt.gpu >= 0 then
		print("Running on GPU...")
		require 'cutorch'
		require 'cunn'
	end
	-- Load models
	encoder = torch.load(opt.loadfile .. 'encoder.t7')
	decoder = torch.load(opt.loadfile .. 'decoder.t7')
	print("Models loaded!")
	-- Load data
	local train_data = data.new(opt, opt.data_file)
	local valid_data = data.new(opt, opt.val_data_file)
	print("Data loaded!")
	-- Check/save results
	encodeDecode(train_data, encoder, decoder, 'enc_ptb_results_train.hdf5')
	encodeDecode(valid_data, encoder, decoder, 'enc_ptb_results_valid.hdf5')
end

main()
