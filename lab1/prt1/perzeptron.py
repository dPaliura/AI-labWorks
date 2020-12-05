from numpy import array
from numpy import concatenate as c

from prt1.neuron import Neuron


class Perzeptron(Neuron):
	train_accuracy_threshold = 0

	def __init__(self, inputs_num, activ_f="sigmoid", activ_p=1, shift=False, weights=None):
		Neuron.__init__(self, 
						inputs_num=inputs_num,
						activ_f=activ_f,
						activ_p=activ_p,
						shift=shift,
						weights=weights)

	def print_weights(self):
		print("Weights of perzeptron:")
		print(self._weights)


	def train1(self, X, d, eps=1e-1, train_rate=None, maxiters=1e4, prints=False):
		if eps <= Perzeptron.train_accuracy_threshold:
			raise Exception("Accuracy parameter eps must be greater then %d" %
							Perzeptron.train_accuracy_threshold)

		def print_(*args, **kwargs):
			if prints:
				print(*args, **kwargs)

		print_("Training parameters:\n",
				"accuracy is ", eps, '\n',
				"training rate is ", train_rate, '\n',
				"Weights before train:\n",
				self._weights,
				sep='')
		print_("\nNow training...\n")
		
		X = c(([1], array(X))) if self._shift else array(X)

		for i in range(0, int(maxiters)):
			y = self._process_input(X)
			delta = (d - y) * (1/y if train_rate is None else train_rate)

			print_("Iteration", i)
			print_("Y_i:")
			print_(y)
			print_("Delta:")
			print_(abs(delta))
			print_("\n")

			if abs(delta) < eps:
				print_("Train stoped after %d iteration" % i)
				print_("Weights after train:\n", self._weights, sep='')
				return delta
			else:
				self._weights = self._weights + X * delta


		print_("Training process reached maxiters=%d number of iterations. Training stoped" % maxiters)
		print_("Weights after train:\n", self._weights, sep='')
		return abs(delta)
