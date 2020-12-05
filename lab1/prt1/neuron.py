from numpy import array
from numpy import concatenate as c
from numpy.random import seed, uniform

from prt1.input_funcs import get_input_fun
from prt1.activ_funcs import get_activ_fun


class Neuron:
    random_seed = 42

    def __init__(self, inputs_num, input_f="sum", activ_f="sigmoid", activ_p=1, shift=False, weights=None):
        if not isinstance(inputs_num, int) and not isinstance(inputs_num, float):
            raise Exception("'inputs_num' must be object of class int or float")
        if inputs_num < 1 or inputs_num % 1 != 0:
            raise Exception("'inputs_num' must be positive finite integer")
        self._inputs_num = inputs_num
        self._shift = not not shift
        if weights is None:
            self._set_random_weights()
        else:
            weights = array(weights)
            if len(weights) != self._inputs_num+int(self._shift):
                raise Exception("'weights' must be list of length equal to %d+%d" % (self._inputs_num, self._shift))
            else:
                self._weights = array(weights)

        self._input_f = get_input_fun(input_f)
        self._activ_f = get_activ_fun(activ_f, activ_p)


    def get_inputs_num(self):
        return int(self._inputs_num)


    def _set_random_weights(self, bipolar=True):
        seed(Neuron.random_seed)
        self._weights = uniform(-1*bipolar, 1, self._inputs_num + int(self._shift))


    def _process_input(self, X):
        return self._activ_f(self._input_f(self._weights * X))


    def process_input(self, X):
        X = array(X)
        if len(X) != self._inputs_num:
            raise Exception("vector 'X' must be list of length equal to inputs_num=%d" % self._inputs_num)

        return self._process_input(c(([1], X)) if self._shift else X)
