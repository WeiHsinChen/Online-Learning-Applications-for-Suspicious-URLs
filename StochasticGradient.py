

class SG:
	def calculate(self, gamma, days, data): 
		if days == 0:
			return None

		import numpy as np
		from scipy.stats import logistic

		# nEx: # of Examples, nF: # of features
		(nEx, nF) = np.shape(data[0]['data'])

		cumuErrorLs = []
		cumuFNLs = []
		cumuTotal = 0
		cumuFalse = 0
		cumuFalseNegative = 0

		w = np.matrix(np.random.rand(nF, 1))
		for d in xrange(days):
			X, Y = data[d]['data'], data[d]['labels']
			
			# Start PA
			(nEx, nF) = np.shape(X)
			for i in xrange(nEx):
				x = X[i, :].T
				y = Y[:, i]

				#  Calculate Error
				probOfPositive = logistic.cdf(w.T * x)
				if probOfPositive >= 0.5:
					predict = 1
				else:
					predict = 0

				if predict != y:
					cumuFalse += 1
					if y == 1:
						cumuFalseNegative += 1
				cumuTotal += 1
				# update w
				w = w + gamma * x * ((y+1/2) - probOfPositive)

			cumuErrorLs.append(cumuFalse/cumuTotal)
			cumuFNLs.append(cumuFalseNegative/cumuTotal)

		return (cumuErrorLs, cumuFNLs)






