import numpy as np
from scipy.stats import logistic
from scipy import sparse
from scipy.sparse import csc_matrix, vstack
# x is k features by N instances
# y is 1 label in {0,1} by N instances
# w1: weight matrix 'w1' between layer 0 and 1
# w2: weight matrix 'w2' between layer 1 and 2
# num_neuron: # of neuron in the layer 1
# gamma: step size for update


class NeuronNetwork:
	# w = csc_matrix(np.random.rand(k, 1))
	def __init__(self, w1=None, w2=None, num_neuron=20 , gamma=0.01):
		self.w1 = w1
		self.w2 = w2
		self.num_neuron = num_neuron
		self.gamma = gamma


	def run(self, x, y):
		print ('start running')
		# N: # of Examples, k: # of features  
		(N, k) = x.shape
		cumu_false = 0.0
		cumu_false_negative = 0.0
		if self.w1 is None:
			self.w1 = csc_matrix(np.random.rand(k+1, self.num_neuron)*0.00001)
		if self.w2 is None:
			self.w2 = csc_matrix(np.random.rand(self.num_neuron+1, 1)*0.00001)

		print ('start for loop')
		# Start NN
		for i in range(N):
			if i % 5 == 0:
				print ('step: ', i)
				print ('Cumulative Error Rate', cumu_false / (i + 1))
				print ('Cumulative False Negative Rate', cumu_false_negative / (i + 1))
			xi = x[i, :].T
			yi = y[i, :][0]

			layer0 = csc_matrix(vstack([csc_matrix([1]), xi]))
			score1 = (self.w1.T).dot(layer0)
			layer1 = csc_matrix(vstack([csc_matrix([1]), np.tanh(score1)]))
			score2 = (self.w2.T).dot(layer1)
			predict = 1 if np.sign(score2) >= 0 else -1

			mistake = 1 if (predict != yi) else 0
			false_negative = 1 if yi == 1 and mistake else 0
			cumu_false += mistake
			cumu_false_negative += false_negative
			
			# calculate sigma for update w1 and w2
			sigma2 = -2 * (yi-score2[0,0]) 
			sigma1 = sigma2 * self.w2[1:, :].multiply(csc_matrix(np.ones((self.num_neuron, 1))) - np.tan(score1).multiply(np.tan(score1))) 

			# update w2
			self.w2 = self.w2 - self.gamma * sigma2 * layer1

			# update w1
			# Method I
			# for j in range(0, self.num_neuron):
			# 	print ('updating num_neuron: ', j)
			# 	self.w1[:, j] = self.w1[:, j] - self.gamma * sigma1[j, 0] * layer0

			# Method II
			for j in range(0, self.num_neuron):
				self.w1[0, j] = self.w1[0, j] - self.gamma * sigma1[j, 0] 
				tmp = sparse.find(xi)
				for k in range(len(tmp[0])):
					idx = tmp[0][k]
					self.w1[idx+1, j] = self.w1[idx+1, j] - self.gamma * sigma1[j, 0] * tmp[2][k]
	
		return (cumu_false, cumu_false_negative)
