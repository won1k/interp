require 'hdf5';

local data = torch.class('data')

function data:__init(data_file, tag_file)
  local f = hdf5.open(data_file, 'r')
  local g
  if tag_file then
    g = hdf5.open(tag_file, 'r')
  else
    g = f
  end
  self.input = {}
  self.output = {}
  self.lengths = f:read('sent_lens'):all():long()
  self.nsent = f:read('nsent'):all():long()
  self.nlengths = self.lengths:size(1)
  self.nclasses = g:read('nclasses_' .. opt.task):all():long()[1]
  self.state_dim = f:read('state_dim'):all():long()[1]
  -- Load sequencer data from total x 650 state file
  local curr_idx = 1
  local states = f:read('states2'):all()
  for i = 1, self.nlengths do
    local len = self.lengths[i]
    local pad_len = len
    local nsent = self.nsent[i]
    self.output[len] = g:read(tostring(len) .. opt.task):all():double()
    if opt.wide > 0 then
      pad_len = len + 2 * torch.floor(opt.dwin/2)
    end
    self.input[len] = torch.Tensor(nsent, pad_len, self.state_dim)
    for j = 1, nsent do
      for k = 1, pad_len do
        self.input[len][j][k] = states[curr_idx]
        curr_idx = curr_idx + 1
      end
    end
    if opt.gpu > 0 then
      require 'cutorch';
      require 'cunn';
      self.input[len] = self.input[len]:cuda()
      self.output[len] = self.output[len]:cuda()
   end
  end
  f:close()
  if tag_file then
   g:close()
  end
end

-- Indexing function
-- Returns table of nsent x sentlen tensors {input, output}
function data.__index(self, idx)
   local input, output
   if type(idx) == "string" then
      return data[idx]
   else
      input = self.input[idx]
      output = self.output[idx]
   end
   return {input, output}
end
