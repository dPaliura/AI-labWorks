from numpy import array, dot, exp, transpose as t, round
from numpy.random import random


class TLP:
    @staticmethod
    def sigmoid(X):
        return 1/(1+exp(-X))
    

    @staticmethod
    def sigmoid_deriv(X):
        Y = TLP.sigmoid(X)
        return Y*(1-Y)


    def __init__(self, input_size, hidden_size, out_size, bias=False):
        self.bias = bias
        self.biasV = random((1, hidden_size)) if bias else 0
        self.biasW = random((1, out_size)) if bias else 0
        self.V = 2*random((input_size, hidden_size)) - 0.5
        self.W = 2*random((hidden_size, out_size)) - 0.5
        self.n = input_size
        self.p = hidden_size
        self.m = out_size
    

    def _feed_forward(self, X):
        Z = self.sigmoid(dot(X, self.V) + self.biasV)
        return self.sigmoid(dot(Z, self.W) + self.biasW)


    def feed_forward(self, X, show_progress=False):
        if X.shape[0] == 1:
            return self._feed_forward(X)
        
        N = X.shape[0]
        Y = []
        if show_progress:
            progress = -1
        for i in range(N):
            Y.append(self._feed_forward(X[i]))
            if show_progress:    
                new_progress = round(100*(i+1)/N)
                if new_progress > progress:
                    progress = new_progress
                    print("{}%".format(progress))
        return array(Y)

        


    def back_propagation(self, X, Y, train_rate=2):
        X = X.reshape(1, self.n)
        Z_in = dot(X, self.V) + self.biasV
        Z = self.sigmoid(Z_in)
        Y_in = dot(Z, self.W) + self.biasW
        Y_web = self.sigmoid(Y_in)

        '''print("X\n", X)
        print("Z_in\n", Z_in)
        print("Z\n", Z)
        print("Y_in\n", Y_in)
        print("Y_web\n", Y_web)'''
        
        err = Y - Y_web
        sigma = err * (Y_web*(1-Y_web))
        dW = train_rate*dot(Z.T, sigma)
        dbiasW = train_rate*sigma if self.bias else 0
        '''print("err\n", err)
        print("sigma\n", sigma)
        print("dW\n", dW)'''
        
        sigma = dot(sigma, self.W.T) * (Z*(1-Z))
        dV = train_rate*dot(X.T, sigma)
        dbiasV = train_rate*sigma if self.bias else 0

        '''print("sigma\n", sigma)
        print("dV\n", dV)'''
        
        self.V += dV
        self.W += dW
        
        self.biasV += dbiasV
        self.biasW += dbiasW
    

    def back_prop_epoch(self, X, Y, train_rate=2, show_progress=False):
        N = X.shape[0]
        if show_progress:
            progress = -1
        for i in range(N):
            self.back_propagation(X[i], Y[i], train_rate)
            if show_progress:    
                new_progress = round(100*(i+1)/N)
                if new_progress > progress:
                    progress = new_progress
                    print("{}%".format(progress))





