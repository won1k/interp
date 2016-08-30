-- General-purposes training for models of form:
-- LookupTable --> SplitTable --> Sequencer --> LogSoftMax
--
-- Assume train_data, test_data in data.lua format with indexing
-- data[length] = { input (nsent x length), output (nsent x length) }

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

function train(train_data, test_data, model, criterion)
  local last_score = 1e9
  local t = 1
  local params, gradParams = model:getParameters()
  -- Initialize tensors
  local gradDenom = torch.ones(gradParams:size())
  local gradPrevDenom = torch.zeros(gradParams:size())
  local prevGrad = torch.zeros(gradParams:size())
  if opt.gpu > 0 then
    gradDenom = gradDenom:cuda()
    gradPrevDenom = gradPrevDenom:cuda()
    prevGrad = prevGrad:cuda()
  end
  params:uniform(-opt.param_init, opt.param_init)
  while not stop do
    model:training()
    print("Training epoch: " .. t)
    -- Convergence condition
    if opt.epochs > 0 then
      if t >= opt.epochs then
        stop = true
      end
    else
      if opt.learning_rate < opt.min_learning_rate then
        stop = true
      end
    end
    -- SGD loop
    for i = 1, train_data.nlengths do
      local sentlen = train_data.lengths[i]
      local paddedlen = sentlen
      if opt.wide > 0 then
        paddedlen = sentlen + 2 * torch.floor(opt.dwin/2)
      end
      if paddedlen >= opt.dwin then
        print(sentlen)
        local d = train_data[sentlen]
        local input, output = d[1], d[2]
        local nsent = input:size(1)
        for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
          local batch_idx = (sent_idx - 1) * opt.bsize
          local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
          local train_input_mb = input[{{ batch_idx + 1, batch_idx + batch_size }}]
          local train_output_mb = output[{
            { batch_idx + 1, batch_idx + batch_size },
            { torch.floor(opt.dwin/2) + 1, torch.floor(opt.dwin/2) + sentlen }}]
          train_output_mb = nn.SplitTable(2):forward(train_output_mb)
          -- Forward prop
          criterion:forward(model:forward(train_input_mb), train_output_mb)
          model:zeroGradParameters()
          -- Backward prop
          model:backward(train_input_mb, criterion:backward(model.output, train_output_mb))
          -- Adaptive gradient
          gradParams, gradDenom, gradPrevDenom = adaptiveGradient(params, gradParams, gradDenom, gradPrevDenom, prevGrad, opt.adapt)
          -- Parameter update
          params:addcdiv(-opt.learning_rate, gradParams, gradDenom)
          prevGrad:mul(0.9):addcdiv(0.1, gradParams, gradDenom)
        end
      end
      model:forget()
    end
    -- Validation error
    local score = eval(test_data, model, criterion)
    local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, t, score)
    -- Save at end
    if t == opt.epochs then
      torch.save(savefile, model)
      print('saving checkpoint to ' .. savefile)
    end
    -- Learning rate update
    if opt.adapt == 'none' then
      if score > last_score - .0001 then
        opt.learning_rate = opt.learning_rate / 2
      end
    end
    last_score = score
    -- Epoch summary
    print(t, score, opt.learning_rate)
    t = t + 1
  end
end

function eval(data, model, criterion)
  model:evaluate()
  local nll = 0
  local total = 0
  for i = 1, data.nlengths do
    local sentlen = data.lengths[i]
    local d = data[sentlen]
    local nsent = d[1]:size(1)
    local start = opt.dwin
    if opt.wide > 0 then
      start = 0
    end
    for sent_idx = 1, torch.ceil(nsent / opt.bsize) do
      if sentlen > start then
        local batch_idx = (sent_idx - 1) * opt.bsize
        local batch_size = math.min(sent_idx * opt.bsize, nsent) - batch_idx
        local test_input_mb = d[1][{{ batch_idx + 1, batch_idx + batch_size }}]
        local test_output_mb = d[2][{
          { batch_idx + 1, batch_idx + batch_size },
          { torch.floor(opt.dwin/2) + 1, torch.floor(opt.dwin/2) + sentlen }}]
        test_output_mb = nn.SplitTable(2):forward(test_output_mb)

        nll = nll + criterion:forward(model:forward(test_input_mb), test_output_mb) * batch_size
        total = total + sentlen * batch_size
      end
    end
    model:forget()
  end
  local valid = math.exp(nll / total)
  return valid
end

function predict(data, model)
  model:evaluate()
  local output = hdf5.open(opt.testoutfile, 'w')
  local accuracy = 0
  local total = 0
  local start = opt.dwin
  local lengths = {}
  if opt.wide > 0 then
    start = 0
  end
  for i = 1, data.nlengths do
    local sentlen = data.lengths[i]
    if sentlen > start then
      table.insert(lengths, sentlen)
      local test_input = data[sentlen][1] -- nsent x senquence_len tensor
      local test_output = data[sentlen][2][{{},
        { torch.floor(opt.dwin/2) + 1, torch.floor(opt.dwin/2) + sentlen }}]
      local test_pred = model:forward(test_input)
      local maxidx = {}
      for j = 1, #test_pred do
        _, maxidx[j] = test_pred[j]:max(2)
      end
      maxidx = nn.JoinTable(2):forward(maxidx)
      output:write(tostring(sentlen), maxidx:long())
      output:write(tostring(sentlen) .. '_target', test_output:long())
      accuracy = accuracy + torch.eq(maxidx:long(), test_output:long()):sum()
      total = total + test_output:long():ge(0):sum()
    end
  end
  output:write('dwin', torch.Tensor{opt.dwin}:long())
  output:write('sent_lens', torch.Tensor(lengths):long())
  accuracy = accuracy / total
  output:write('accuracy', torch.Tensor{accuracy}:double())
  output:close()
  print('Accuracy', accuracy)
end
