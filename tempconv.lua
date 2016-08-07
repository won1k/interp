-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'checkpoint/lstm_states.h5', 'data file')
cmd:option('-tagfile', 'convert/train_parsed_chunks.hdf5', 'chunking tag file')
cmd:option('-savefile', 'tempconv_model.t7', 'file to save modelwork')
cmd:option('-testfile', 'checkpoint/lstm_states_test.h5', 'lstm states for test')
cmd:option('-testtagfile', 'convert/test_parsed_chunks.hdf5', 'chunking tag file for test')
cmd:option('-testoutfile', 'test_results.hdf5', 'output file for test')
cmd:option('-gpu', 0, 'whether to use gpu')

-- Hyperparameters
cmd:option('-state_dim', 650, 'LSTM state dimension')
cmd:option('-lambda', 1, 'learning rate')
cmd:option('-epochs', 30, 'epochs')
cmd:option('-bsize', 32, 'mini-batch size')
cmd:option('-dhid', 300, 'hidden dimension')
cmd:option('-dwin', 5, 'window size')

local data = torch.class('data')
function data:__init(data_file, tag_file)
   local f = hdf5.open(data_file, 'r')
	 local g = hdf5.open(tag_file, 'r')
	 self.input = f:read('states2'):all():double()
	 self.output = g:read('tags'):all():long()
   self.nclasses = g:read('nclasses'):all():long()[1]
	 self.length = self.input:size(1)
	 self.state_dim = self.input:size(2)
	 if opt.gpu > 0 then
		 self.input:cuda()
		 self.output:cuda()
	 end
   f:close()
	 g:close()
end

function make_model(data)
	-- Initialize modelwork
	local model = nn.Sequential() -- bsize x (state_dim*dwin)
	model:add(nn.TemporalConvolution(data.state_dim, opt.dhid, opt.dwin))
	model:add(nn.HardTanh())
	model:add(nn.Linear(opt.dhid, data.nclasses)) -- bsize x nclasses scores
	model:add(nn.LogSoftMax()) -- bsize x nclasses log probs.

	-- Define criterion, initialize parameters
	local criterion = nn.ClassNLLCriterion()

	if opt.gpu > 0 then
		model:cuda()
		criterion:cuda()
	end

	return model, criterion
end

function train(train_data, test_data, model, criterion)
	local last_score = 1e9
	local n = train_input:size()[1]
	local params, gradParams = model:getParameters()
  params:uniform(-opt.param_init, opt.param_init)

	-- Train model
	for t = 1, epochs do
		print("Training epoch: " .. t)
		for idx = 1, torch.floor(n / bsize) do
			-- Create minibatches
			local start_idx = (idx-1) * bsize
			local mb_size = math.min(idx * bsize, n) - start_idx
			local train_input_mb = train_data.input[{{ start_idx + 1, start_idx + mb_size }}] -- to make sure enough space for convolution
			local train_output_mb = train_data.output[{{ start_idx + 1, start_idx + mb_size }}]
			train_input_mb = temp:forward(train_input_mb):clone()
			train_output_mb = train_output_mb[{{ torch.floor(dwin/2) + 1, mb_size - torch.floor(dwin/2) }}]

			-- Manual SGD
			criterion:forward(model:forward(train_input_mb), train_output_mb)
			model:zeroGradParameters()
			model:backward(train_input_mb, criterion:backward(model.output, train_output_mb))
			model:updateParameters(lambda)
		end

		print("Testing...")
		-- Validation
		local score = eval(test_windowed, test_output_windowed, model)
		print("Validation accuracy", val_acc)

		if score > last_score - .3 then
       opt.learning_rate = opt.learning_rate / 2
			 print("Learning rate", opt.learning_rate)
    end
    last_score = score
	end

	-- Save model
	local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, t, last_score)
	torch.save(savefile, model)

	return model
end

function eval(test_data, model)
	local test_pred = model:forward(test_data.input)
	local n = test_pred:size(1)
	local maxval, maxidx = test_pred:max(2)
	maxidx = maxidx:squeeze()
	local accuracy = 0
	for i = 1, n do
		if maxidx[i] == test_data.output[i] then
			accuracy = accuracy + 1
		end
	end
	return accuracy/n
end

function test(test_data, model, criterion)
	local test_pred = model:forward(test_data.input)
	local test_output = test_data.output[{{1, test_pred:size(1)}}]
	local maxval, maxidx = test_pred:max(2)
	maxidx = maxidx:squeeze()
	local output = hdf5.open(opt.testoutfile, 'w')
	output:write('predictions', test_pred:float())
	output:write('output', maxidx:long())
	output:write('tags', test_output:float())
	output:write('dwin', torch.Tensor{dwin}:long())
	output:close()
end

function main()
   	-- Parse input params
   	opt = cmd:parse(arg)
		if opt.gpu > 0 then
			require 'cutorch';
			require 'cunn';
			require 'cudnn';
		end

    -- Load training data
    local train_data = data(opt.datafile, opt.tagfile)
		local test_data = data(opt.testfile, opt.testtagfile)

		-- Make model
		local model, criterion = make_model(train_data)

    -- Train.
		model = train(train_data, test_data, model, criterion)

    -- Test.
		test(test_data, model, criterion)
end

main()
