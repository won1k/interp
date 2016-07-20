-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")
require 'cudnn';

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'convert_word/data.hdf5', 'data file')
cmd:option('-tagfile', 'convert_word/train_parsed_chunks.hdf5', 'chunking tag file')
cmd:option('-networkfile', 'tempconv_word_network.t7', 'file to save network')
cmd:option('-testfile', 'convert_word/dataval.h5', 'lstm states for test')
cmd:option('-testoutfile', 'word_test_results.hdf5', 'output file for test')
cmd:option('-ltweights', 'checkpoint/lstm_LT.h5', 'file containing LT weights')
cmd:option('-gpu', 0, 'whether to use gpu')

-- Hyperparameters
cmd:option('-state_dim', 650, 'word embedding dimension')
cmd:option('-lambda', 0.01, 'learning rate')
cmd:option('-epochs', 30, 'epochs')
cmd:option('-bsize', 32, 'mini-batch size')
cmd:option('-dhid', 300, 'hidden dimension')
cmd:option('-dwin', 5, 'window size')

-- Neural network model (no pretrained vectors)
function NN(train_input, train_output, lt_weights, dwin, state_dim, lambda, epochs, bsize, dhid, gpu)
	local n = train_input:size()[1]

	-- Initialize network
	local net = nn.Sequential() -- bsize x (state_dim*dwin)
	net:add(nn.Linear(state_dim*dwin, dhid)) -- (bsize - dwin + 1) x dhid
	net:add(nn.HardTanh())
	net:add(nn.Linear(dhid, nclasses)) -- (bsize - dwin + 1) x nclasses scores
	net:add(nn.LogSoftMax()) -- (bsize - dwin + 1) x nclasses log probs.

	-- Initialize LT with pretrained weights
	local LT = nn.LookupTable(nfeatures, state_dim)
	LT.weight = lt_weights

	-- Initialize temp. convolution layer to get windows
	local temp = nn.TemporalConvolution(state_dim, state_dim*dwin, dwin)
	temp.bias = torch.zeros(state_dim*dwin)
	temp.weight = torch.eye(state_dim*dwin)

	-- Define criterion, initialize parameters
	local criterion = nn.ClassNLLCriterion()

	if gpu > 0 then
		net:cuda()
		LT:cuda()
		temp:cuda()
		criterion:cuda()
	end

	-- Train network
	for t = 1, epochs do
		print("Training epoch: " .. t)
		for idx = 1, torch.floor(n / bsize) do
			-- Create minibatches
			local start_idx = (idx-1) * bsize
			print(start_idx)
			local mb_size = math.min(idx * bsize, n) - start_idx
			local train_input_mb = train_input[{{ start_idx + 1, start_idx + mb_size }}] -- to make sure enough space for convolution
			local train_output_mb = train_output[{{ start_idx + 1, start_idx + mb_size }}]
			train_input_mb = temp:forward(LT:forward(train_input_mb))
			train_output_mb = train_output_mb[{{ torch.floor(dwin/2) + 1, mb_size - torch.floor(dwin/2) }}]

			-- Manual SGD
			criterion:forward(net:forward(train_input_mb), train_output_mb)
			net:zeroGradParameters()
			net:backward(train_input_mb, criterion:backward(net.output, train_output_mb))
			net:updateParameters(lambda)
		end
	end

	return net
end

function main()
   	-- Parse input params
   	opt = cmd:parse(arg)
   	local f = hdf5.open(opt.datafile, 'r')
		local g = hdf5.open(opt.tagfile, 'r')

    -- Parse hyperparameters
    local state_dim = opt.state_dim
    local lambda = opt.lambda
    local epochs = opt.epochs
    local bsize = opt.bsize
    local dhid = opt.dhid
		local dwin = opt.dwin
		local gpu = opt.gpu

    -- Load training data
    local train_input = f:read('target'):all():long()
    local train_output = g:read('chunks'):all():long()
		if gpu > 0 then
			train_input = train_input:cuda()
	    train_output = train_output:cuda()
		nfeatures = f:read('nfeatures'):all():long()[1]
		nclasses = g:read('nclasses'):all():long()[1]

		-- Load LT weights
		local h = hdf5.open(opt.ltweights, 'r')
		local lt_weights = h:read('weights'):all():double()

    -- Train.
    local net = NN(train_input, train_output, lt_weights, dwin, state_dim, lambda, epochs, bsize, dhid, gpu)
		torch.save(opt.networkfile, {dwin = dwin, nclasses = nclasses, state_dim = state_dim, network = net})

    -- Test.
		local h = hdf5.open(opt.testfile, 'r')
		local LT = nn.LookupTable(nfeatures, state_dim)
		LT.weight = lt_weights
		local temp = nn.TemporalConvolution(state_dim, state_dim*dwin, dwin)
		temp.bias = torch.zeros(state_dim*dwin)
		temp.weight = torch.eye(state_dim*dwin)
		test_input = temp:forward(LT:forward(h:read('target'):all():long()))
		local test_pred = net:forward(test_input)
		local maxval, maxidx = test_pred:max(2)
		maxidx = maxidx:squeeze()
		local output = hdf5.open(opt.testoutfile, 'w')
		output:write('predictions', test_pred)
		output:write('output', maxidx)
		output:write('dwin', torch.Tensor{dwin})
end

main()
