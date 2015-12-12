

class SVM:
	def train(self, x, y, param): 
		"""
		Training examples and labels in day i ~ j
		:type x: List[List[int]]
		:type y: List[int]
		:rtype: libsvm.model

		TODO: 
			1. transverse sparse input into
				x = [[example 1], [example 2], ..., [example n]]
				y = [label 1, label 2, ... label n]
			2. param = svm_parameter('-t 0 -c 100 -b 1')
		"""

		from libsvm.python.svmutil import *

		prob  = svm_problem(y, x)
		model = svm_train(prob, param)

		return model

	def test(self, model, x, y):
		"""
		Testing examples and labels in a single day i
		:type x: List[List[int]]
		:type y: List[int]
		:rtype: (List[int], (float, float, float), List[float])

		TODO: 
			1. transverse sparse input into
				x = [[example 1], [example 2], ..., [example n]]
				y = [label 1, label 2, ... label n]

			2. change the path of libsvm
		"""

		from libsvm.python.svmutil import *

		# p_label: predicted lables
		# p_acc: accuracy rate
		# p_val: probability estimates
		p_label, p_acc, p_val = svm_predict(y, x, model, "-b 1")

		return (p_label, p_acc, p_val)

