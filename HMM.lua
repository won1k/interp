require 'nn'
require 'Utils.lua'

local HMM, parent = torch.class('nn.HMM', 'nn.Module')

function HMM:__init(n_states, n_obs)
	parent.__init(self)

	self.n_states = n_states
	self.n_obs = n_obs
	-- weights is single n_states x (1 + n_states + n_obs) tensor:
	-- first n_states x 1 is initial distribution
	-- next n_states x n_states is transition matrix
	-- final n_states x n_obs is emission matrix
	self.weight = torch.ones(n_states, 1 + n_states + n_obs)
	self.gradWeight = torch.zeros(n_states, 1 + n_states + n_obs)
	self.gradInput = torch.Tensor()
	self.output = torch.Tensor()

	-- Matrices for easier computations
	self:updateMatrices()
end

function HMM:updateMatrices()
	-- Normalize weights
	self.weight:renorm(1, 2, 1)
	for i = 1, self.n_states do
		self.weight[{{i},{2, 1 + self.n_states}}] = self.weight[{{i},{2, 1 + self.n_states}}] / self.weight[{{i}, {2, 1 + self.n_states}}]:sum()
		self.weight[{{i},{2 + self.n_states, self.weight:size(2)}}] = self.weight[{{i},{2 + self.n_states, self.weight:size(2)}}] / 
			self.weight[{{i},{2 + self.n_states, self.weight:size(2)}}]:sum()
	end
	self._initProbs = self.weight[{{},{1}}]:squeeze()
	self._transProbs = self.weight[{{},{2, 1 + self.n_states}}]
	self._emitProbs = self.weight[{{},{2 + self.n_states, 1 + self.n_states + self.n_obs}}]
end

function HMM:simulate(n_steps)
	local T = n_steps or 10
	local states = torch.Tensor(T)
	local observations = torch.Tensor(T)
	states[1] = torch.multinomial(self._initProbs, 1, true)
	for t = 1, T-1 do
		observations[t] = torch.multinomial(self._emitProbs[states[t]], 1, true) 
		states[t+1] = torch.multinomial(self._transProbs[states[t]], 1, true)
	end
	observations[T] = torch.multinomial(self._emitProbs[states[T-1]], 1, true)
	return states, observations
end

function HMM:viterbi(observations)
	-- Assume observations is T-vector
	local T = observations:size(1)
	local maxProb = torch.zeros(self.n_states, T)
	local maxIdx = torch.zeros(self.n_states, T)
	local maxSeq = torch.zeros(T)
	-- Initialization
	for i = 1, self.n_states do
		maxProb[i][1] = self._initProbs[i] * self._emitProbs[i][observations[1]]
		maxIdx[i][1] = 0
	end
	-- Recursion
	for t = 2, T do
		for i = 1, self.n_states do
			local maxPrevProb, maxPrevIdx = maxProb[{{},{t-1}}]:squeeze():max(1)
			maxProb[i][t] = maxPrevProb:squeeze() * self._emitProbs[i][observations[t]]
			maxIdx[i][t] = maxPrevIdx:squeeze()
		end
	end
	-- Termination
	local maxFinalProb, maxFinalIdx = maxProb[{{},{T}}]:squeeze():max(1)
	maxSeq[T] = maxFinalIdx:squeeze()
	-- Path-finding
	for t = T-1, 1, -1 do
		maxSeq[t] = maxIdx[maxSeq[t+1]][t+1]
	end
	return maxSeq, maxFinalProb:squeeze()
end

function HMM:fb(observations)
	local T = observations:size(1)
	local alpha = torch.zeros(self.n_states, T)
	local beta = torch.zeros(self.n_states, T)
	-- Initialization
	for i = 1, self.n_states do
		alpha[i][1] = self._initProbs[i] * self._emitProbs[i][observations[1]]
		beta[i][T] = 1
	end
	-- Recursion
	for t = 1, T-1 do
		for i = 1, self.n_states do
			alpha[i][t+1] = torch.cmul(self._transProbs[{{},{i}}], alpha[{{},{t}}]):sum() * self._emitProbs[i][observations[t+1]]
			beta[i][T-t] = torch.cmul(self._transProbs[i]:cmul(self._emitProbs[{{},{observations[T-t+1]}}]), beta[{{},{T-t+1}}])
		end
	end
	return alpha, beta
end

function HMM:baumwelch(observations, n_iter)
	local T = observations:size(1)
	local alpha, beta
	local gamma = torch.zeros(self.n_states, T)
	local eta = torch.zeros(T-1, self.n_states, self.n_states)
	-- Training loop
	for iter = 1, n_iter do
		-- Forward-backward
		alpha, beta = self:fb(observations)
		-- Compute count tensors
		gamma = torch.cmul(alpha, beta)
		gamma:cdiv(gamma:sum(1):expandAs(gamma)) -- Normalization
		for t = 1, T-1 do
			eta[t] = torch.cmul(
				torch.cmul(self._transProbs, self._emitProbs[{{},{observations[t+1]}}]:transpose(1,2):expandAs(self._transProbs)),
				torch.cmul(alpha[{{},{t}}]:expandAs(self._transProbs), beta[{{},{t+1}}]:transpose(1,2):expandAs(self._transProbs))
			)
			eta[t]:cdiv(torch.Tensor(eta[t]:size()):fill(eta[t]:sum())) -- Normalization
		end
		-- Update weights
		self.weight[{{},{1}}] = gamma[{{},{1}}]
		self.weight[{{},{2, 1 + self.n_states}}] = eta:sum(1):squeeze():cdiv(gamma:sum(2):expandAs(eta:sum(1):squeeze()))
		for k = 1, self.n_obs do
			local observations_ind = torch.zeros(1,T)
			for t = 1, T do
				if observations[t] == k then
					observations_ind[1][t] = 1
				end
			end
			self.weight[{{},{2 + self.n_states, 1 + self.n_states + k}}] = torch.cdiv(
				torch.cmul(gamma, observations_ind:expandAs(gamma)):sum(2),
				gamma:sum(2))
		end
		self:updateMatrices()
	end
end

-- Input = observations (n_batch x length)
-- Output = probability distribution over all hidden states
-- Loss = cross-entropy over hidden states
function HMM:updateOutput(input)
	local n_batch, length = input:size(1), input:size(2)
	self.output:resize(n_batch, length, self.n_states)
	-- Forward-backward to get hidden probabilities
	for b = 1, n_batch do
		local alpha, beta = self:fb(input)
		self.output[b] = torch.cmul(alpha, beta):transpose(1,2)
		self.output[b]:renorm(1, 1, 1)
	end
	return self.output
end

function HMM:updateGradInput(input, gradOutput)

end

function HMM:accGradParameters()
end

function HMM:updateGradTables()
end