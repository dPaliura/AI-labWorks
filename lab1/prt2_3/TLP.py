from numpy import array
from numpy import concatenate as c
from numpy.random import seed, uniform
from prt2_3.my_funcs import sigmoid, sigmoid_deriv


class TwoLayerPerzeptron:
    rnd_seed = 42

    matrix_norma = lambda M: max(c(abs(M)))

    def __init__(self, n, p, m):
        self.n = n
        self.p = p
        self.m = m
        self.V = None
        self.W = None
        self.init_weights()


    def init_weights(self):
        seed(self.rnd_seed)
        self.V = array([uniform(-.5, .5, self.p) for i in range(0, self.n+1)])
        self.W = array([uniform(-.5, .5, self.m) for i in range(0, self.p+1)])


    def print_weights(self):
        print("Weights in hidden layer (size: (1+n) x p):")
        print(self.V)
        print("Weights in output layer (size: (1+p) x m):")
        print(self.W)


    def process_input(self, X, full=False):
        X = array(X)
        n = self.n
        p = self.p
        m = self.m

        Z_in = self.V[0] + \
            array([sum(X * self.V[1:n+1, i]) for i in range(0, p)])
        Z = array(sigmoid(Z_in))

        Y_in = self.W[0] + \
            array([sum(Z * self.W[1:p+1, i]) for i in range(0, m)])
        Y = array(sigmoid(Y_in))

        return {
            "Z_in": Z_in,
            "Z": Z,
            "Y_in": Y_in,
            "Y": Y
        } if full else Y


    def process_inputs(self, X):
        return c([self.process_input(x) for x in X])


    def train1(self, X, T, train_rate=None, prints=False):
        X_ = c(([1], X))
        T = array(T)
        n = self.n
        p = self.p

        def print_(*args, **kwargs):
            print(*args, **kwargs) if prints else None

        #print_("Weights BEFORE train:\n")
        #self.print_weights() if prints else None

        result = self.process_input(X, full=True)
        Z_in = result['Z_in']
        Z = result['Z']
        Z_ = c(([1], Z))
        Y_in = result['Y_in']
        Y = result['Y']

        sigma_Y = (T-Y)*sigmoid_deriv(Y_in)
        delta_w = array([sigma_Y * Z_[i] for i in range(0, 1+p)]) *\
                  ((1/Y) if train_rate is None else train_rate)
        print_("\nDelta W:\n", abs(delta_w))

        sigma_in = array([sum(sigma_Y * self.W[i] for i in range(1, p+1))]).T[0]
        sigma_Z = sigma_in * sigmoid_deriv(Z_in)
        delta_v = array([sigma_Z * X_[i] for i in range(0, 1+n)]) *\
                ((1/Z) if train_rate is None else train_rate)

        #print_("Sigma Z:", sigma_Z)
        print_("Delta V:\n", abs(delta_v), "\n")

        self.W = self.W + delta_w
        self.V = self.V + delta_v

        #print_("Weights AFTER train:\n")
        #self.print_weights() if prints else None

        return delta_w


    def train_online(self, X, T, eps=1e-2, train_rate=None, maxiters=1e6, err_norma=matrix_norma, prints=False, each_iter_print=100):
        N = len(X)
        iteration = 0
        if N != len(T):
            raise Exception("Number of inputs in trainig examples not equal to number of expected outputs")

        def print_(*args, **kwargs):
            if prints:
                print(*args, **kwargs) if not (iteration % each_iter_print) else None

        X = array(X)
        T = array(T)
        maxiters = int(maxiters)

        print_("Training parameters:\n",
               "accuracy is ", eps, '\n',
               "training rate is ", train_rate, '\n',
               "Weights before train:\n")
        self.print_weights() if prints else None
        print_("\nNow training...\n")


        for iteration in range(0, maxiters):
            Y_web = self.process_inputs(X)

            print_("\t\tITERATION", iteration, "\n")
            print_("Y of web BEFORE train:")
            print_(Y_web)
            #print_("Error:")
            #print_(error, '\n')

            for i in range(0, N):
                print_("\tTraining example {}".format(i+1))
                error = self.train1(X[i], T[i], train_rate, prints=not (iteration % each_iter_print))

            Y_web = self.process_inputs(X)
            print_("Y of web AFTER train:")
            print_(Y_web, "\n")

            err_nrm = err_norma(error)
            if err_nrm < eps:
                if prints:
                    print_("Train stoped after %d iteration" % iteration)
                    print_("Weights after train:")
                    self.print_weights()
                return

        iteration = 0

        print_("Training algorithm reached maximum iterations - %d" % maxiters)
        print_("Reached accuracy - %f" % err_nrm)
        print_("Weights after train:")
        self.print_weights()


