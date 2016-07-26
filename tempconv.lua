-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'checkpoint/lstm_states.h5', 'data file')
cmd:option('-tagfile', 'convert/train_parsed_chunks.hdf5', 'chunking tag file')
cmd:option('-networkfile', 'tempconv_network.t7', 'file to save network')
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

function eval(test_input, test_output, net)
	local n = test_input:size(1)
	local test_pred = net:forward(test_input)
	local maxval, maxidx = test_pred:max(2)
	maxidx = maxidx:squeeze()
	local accuracy = 0
	for i = 1, n do
		if maxidx[i] == test_output[i] then
			accuracy = accuracy + 1
		end
	end
	return accuracy/n
end

-- Neural network model
function NN(train_input, train_output, test_input, test_output, dwin, state_dim, lambda, epochs, bsize, dhid)
	local n = train_input:size()[1]

	-- Initialize network
	local net = nn.Sequential() -- bsize x (state_dim*dwin)
	net:add(nn.Linear(state_dim*dwin, dhid)) -- bsize x dhid
	net:add(nn.HardTanh())
	net:add(nn.Linear(dhid, nclasses)) -- bsize x nclasses scores
	net:add(nn.LogSoftMax()) -- bsize x nclasses log probs.

	-- Initialize temp. convolution layer to get windows
	local temp = nn.TemporalConvolution(state_dim, state_dim*dwin, dwin)
	temp.bias = torch.zeros(state_dim*dwin)
	temp.weight = torch.eye(state_dim*dwin)

	-- Define criterion, initialize parameters
	local criterion = nn.ClassNLLCriterion()

	if opt.gpu > 0 then
		net:cuda()
		temp:cuda()
		criterion:cuda()
	end

	local test_windowed = temp:forward(test_input):clone()
	local test_len = test_windowed:size(1)
	local test_output_windowed = test_output[{{torch.floor(dwin/2) + 1, test_len + torch.floor(dwin/2)}}]

	-- Train network
	for t = 1, epochs do
		print("Training epoch: " .. t)
		for idx = 1, torch.floor(n / bsize) do
			-- Create minibatches
			local start_idx = (idx-1) * bsize
			local mb_size = math.min(idx * bsize, n) - start_idx
			local train_input_mb = train_input[{{ start_idx + 1, start_idx + mb_size }}] -- to make sure enough space for convolution
			local train_output_mb = train_output[{{ start_idx + 1, start_idx + mb_size }}]
			train_input_mb = temp:forward(train_input_mb):clone()
			train_output_mb = train_output_mb[{{ torch.floor(dwin/2) + 1, mb_size - torch.floor(dwin/2) }}]

			-- Manual SGD
			criterion:forward(net:forward(train_input_mb), train_output_mb)
			net:zeroGradParameters()
			net:backward(train_input_mb, criterion:backward(net.output, train_output_mb))
			net:updateParameters(lambda)
		end

		print("Testing...")
		-- Validation
		local val_acc = eval(test_windowed, test_output_windowed, net)
		print("Validation accuracy: " .. val_acc)
	end

	return net, test_windowed, test_output_windowed
end

function main()
   	-- Parse input params
   	opt = cmd:parse(arg)
   	local f = hdf5.open(opt.datafile, 'r')
		local g = hdf5.open(opt.tagfile, 'r')
		local h = hdf5.open(opt.testfile, 'r')
		local j = hdf5.open(opt.testtagfile, 'r')

    -- Parse hyperparameters
    local state_dim = opt.state_dim
    local lambda = opt.lambda
    local epochs = opt.epochs
    local bsize = opt.bsize
    local dhid = opt.dhid
		local dwin = opt.dwin

    -- Load training data
    local train_input = f:read('states2'):all():double()
    local train_output = g:read('tags'):all():long()
		local test_input = h:read('states2'):all():double()
		local test_output = j:read('tags'):all():long()
		if opt.gpu > 0 then
			train_input = train_input:cuda()
	    train_output = train_output:cuda()
			test_input = test_input:cuda()
			test_output = test_output:cuda()
		end
		nclasses = g:read('nclasses'):all():long()[1]

    -- Train.
    local net, test_windowed, test_output_windowed = NN(train_input, train_output, test_input, test_output, dwin, state_dim, lambda, epochs, bsize, dhid, gpu)
		torch.save(opt.networkfile, {dwin = dwin, nclasses = nclasses, state_dim = state_dim, network = net})

    -- Test.
		local test_pred = net:forward(test_windowed)
		local maxval, maxidx = test_pred:max(2)
		maxidx = maxidx:squeeze()
		local output = hdf5.open(opt.testoutfile, 'w')
		output:write('predictions', test_pred)
		output:write('output', maxidx)
		output:write('chunks', test_output_windowed)
		output:write('dwin', torch.Tensor{dwin})
		output:close()
end

main()
