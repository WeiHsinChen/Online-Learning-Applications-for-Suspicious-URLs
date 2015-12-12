

class SG:
	def calculate(self, gamma, days, data): 
		if days == 0:
			return None

		import numpy as np
		from scipy.stats import logistic
		from scipy.sparse import csc_matrix

		# nEx: # of Examples, nF: # of features
		(nEx, nF) = np.shape(data[0]['data'])

		cumuErrorLs = []
		cumuFNLs = []
		cumuTotal = 0.0
		cumuFalse = 0.0
		cumuFalseNegative = 0.0

		# w = csc_matrix(np.random.rand(nF, 1)*0.001)
		w = csc_matrix(np.zeros((nF, 1)))
		for d in xrange(days):
			X, Y = data[d]['data'], data[d]['labels']
			print 'day:', d
			# Start PA
			(nEx, nF) = np.shape(X)
			for i in xrange(nEx):
				if i % 100 == 0:
					print 'day: ', d, ', step: ', i
					print 'Cumulative Error Rate', cumuFalse/(cumuTotal+1)
					print 'Cumulative False Negative Rate', cumuFalseNegative/(cumuTotal+1)
				x = X[i, :].T
				y = Y[i, :][0]


				#  Calculate Error
				# print 'w.T', w.T
				# print 'x', x
				# print 'w.T * x', w.T * x


				tmp = (w.T).dot(x)
				probOfPositive = logistic.cdf(tmp[0,0])
				if probOfPositive >= 0.5:
					predict = 1
				else:
					predict = -1

				# print 'predict',  predict
				# print 'y', y

				if predict != y:
					cumuFalse += 1
					if y == 1:
						cumuFalseNegative += 1
				cumuTotal += 1
				# update w
				w = w + gamma * x * ((y+1)/2 - probOfPositive)

			cumuErrorLs.append(cumuFalse/cumuTotal)
			cumuFNLs.append(cumuFalseNegative/cumuTotal)

		return (cumuErrorLs, cumuFNLs)






