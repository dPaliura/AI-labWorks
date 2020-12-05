from math import exp
from numpy import array


def isiterable(x):
    return hasattr(x, '__iter__')


def sigmoid(X):
    f = lambda x: 1/(1+exp(-x))
    return array([f(x) for x in X]) if isiterable(X) else f(X)


def sigmoid_deriv(X):
    f = lambda x: sigmoid(x)*(1-sigmoid(x))
    return array([f(x) for x in X]) if isiterable(X) else f(X)


