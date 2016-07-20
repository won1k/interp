-- Only requirements allowed
require("hdf5")
require("nn")
require("optim")
require 'cudnn';

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-networkfile', 'tempconv_network.t7', 'file to save network')
cmd:option('-testfile', 'checkpoint/lstm_states_test.h5', 'lstm states for test')
cmd:option('-testoutfile', 'test_results.hdf5', 'output file for test')
cmd:option('-gpu', 0, 'whether to use GPU')

-- Hyperparameters
cmd:option('-state_dim', 650, 'LSTM state dimension')
cmd:option('-dwin', 5, 'window size')

function main()
   	-- Parse input params
   	opt = cmd:parse(arg)
   	local f = hdf5.open(opt.testfile, 'r')

    -- Parse hyperparameters
    local state_dim = opt.state_dim
		local dwin = opt.dwin
    local gpu = opt.gpu

    -- Load training data
    local test_input
    if gpu > 0 then
  		test_input = f:read('states2'):all():cuda()
    else
      test_input = f:read('states2'):all():double()
    end

    -- Load networks
    local net = torch.load('tempconv_network.t7').network
    if gpu > 0 then
      net = net:cuda()
    end

    local temp = nn.TemporalConvolution(state_dim, state_dim*dwin, dwin)
    if gpu > 0 then
  	  temp.bias = torch.zeros(state_dim*dwin):cuda()
  	  temp.weight = torch.eye(state_dim*dwin):cuda()
    else
      temp.bias = torch.zeros(state_dim*dwin)
  	  temp.weight = torch.eye(state_dim*dwin)
    end

    -- Test.
		local test_pred = net:forward(temp:forward(test_input))
		local maxval, maxidx = test_pred:max(2)
		maxidx = maxidx:squeeze()
		local output = hdf5.open(opt.testoutfile, 'w')
		output:write('predictions', test_pred)
		output:write('output', maxidx)
		output:write('dwin', torch.Tensor{dwin})
		output:close()
end

main()
