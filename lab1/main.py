from prt1.perzeptron import Perzeptron
from prt2_3.TLP import TwoLayerPerzeptron
from os import system

# Training data for part 1 of laboratory work
#
# N is number of inputs in One-Layer Perzeptron
# X1 is single training example input
# T1 is single training example expected output
# eps1 is accuracy of trainig in part 1
N = 4
X1 = [0.4, 1, 0.5, 0.2]
T1 = 0.5
eps1 = 1e-3

# Training data for laboratory work parts 2 and 3
#
# n, p, m - numbers of neurons on input, hidden and output layers accordingly
# X2 is trainig examples inputs
# T2 is training examples expected outputs
# V is validation sample
# V_T is correct results for validation sample
# eps2 is accuracy of trainig in part 1
# alpha is train rate for back propagation algorithm
n, p, m = (2, 3, 1)
X2 = [[1, 1], [2, 1], [1.5, 1.5], [1, 2.5]]
T2 = [[0.02], [0.03], [0.03], [0.035]]
V = [[1, 1.5]]
V_T = [0.025]
eps2 = 1e-3
maxiters2 = 5e4
alpha = 1


# Training data for laboratory work part 3
#
# n, p, m - numbers of neurons on input, hidden and output layers accordingly
# X2 is trainig examples inputs
# T2 is training examples expected outputs
# V is validation sample
# V_T is correct results for validation sample
# eps2 is accuracy of trainig in part 1
# alpha is train rate for back propagation algorithm
n1, p1, m1 = (1, 1, 1)
X2 = [0.5]
T2 = [0.7]



# Training data for laboratory work part 3
#
# n, p, m - numbers of neurons on input, hidden and output layers accordingly
# X2 is trainig examples inputs
# T2 is training examples expected outputs
# V is validation sample
# V_T is correct results for validation sample
# eps2 is accuracy of trainig in part 1
# alpha is train rate for back propagation algorithm
n2, p2, m2 = (2, 3, 1)
X3 = [[1, 1], [2, 1], [1.5, 1.5], [1, 2.5]]
T3 = [[0.02], [0.03], [0.03], [0.035]]
V = [[1, 1.5]]
V_T = [0.025]
eps3 = 1e-3
maxiters3 = 1e4


if __name__ == '__main__':
    ##
    #   Part 1
    #
    input("Press 'Enter' to reproduce part 1")
    system("cls")
    print("\n\tINITIALIZATION")
    print("Seed for numpy.random is set %d" % Perzeptron.random_seed)
    OLP = Perzeptron(N)

    print("One-Layer Perzeptron (OLP) created with weights:")
    OLP.print_weights()

    print("\n\tTRAINING")

    print("OLP computed BEFORE train: \nexample: - {X} \nresult: {Y} \nexpected: {T}\n".format(
        X=X1, Y=OLP.process_input(X1), T=T1))

    OLP.train1(X1, T1, eps1, prints=True)

    print("\n\tCOMPUTATION")

    print("OLP computed AFTER train: \nexample: - {X} \nresult: {Y} \nexpected: {T}\n".format(
        X=X1, Y=OLP.process_input(X1), T=T1))

    ##
    #   Part 2
    #
    input("Press 'Enter' to reproduce part 2")
    system("cls")
    print("\n\tINITIALIZATION")
    print("Seed for numpy.random is set %d" % Perzeptron.random_seed)
    TLP1 = TwoLayerPerzeptron(n1, p1, m1)

    print("Two-Layer Perzeptron (TLP) created with weights:")
    TLP1.print_weights()

    print("\n\tTRAINING")

    print("TLP computed BEFORE train: \nexample: - {X} \nresult: {Y} \nexpected: {T}\n".format(
        X=X2, Y=TLP1.process_input(X2), T=T2))

    for i in range(1, 11):
        print("Iteration %d" % i)
        TLP1.train1(X1, T1, prints=True)

    print("Weights AFTER train")
    TLP1.print_weights()

    print("\n\tCOMPUTATION")

    print("TLP computed AFTER train: \nexample: - {X} \nresult: {Y} \nexpected: {T}\n".format(
        X=X2, Y=TLP1.process_input(X2), T=T2))
    print("Expermental accuracy: {}\n".format(abs(T2 - TLP1.process_input(X2))))


    ##
    #   Part 3
    #
    input("\nPress 'Enter' to reproduce part 3")
    system("cls")
    print("\n\tINITIALIZATION")
    print("Seed for numpy.random is set %d" % TwoLayerPerzeptron.rnd_seed)
    print("Initialization parameters of Two-Layer Perzeptron (TLP):",
          "number of input neurons  n = %d" % n,
          "number of hidden neurons p = %d" % p,
          "number of output neurons m = %d\n" % m,
          sep='\n')
    TLP2 = TwoLayerPerzeptron(n2, p2, m2)

    print("TLP created with weights:")
    TLP2.print_weights()

    print("\n\tTRAINING")

    print("TLP computed BEFORE train: \nexample: - {X} \nresult: {Y} \ncorrect: {T}".format(
        X=V, Y=TLP2.process_inputs(V), T=V_T))
    print("Training set:\ninputs")
    print(X3)
    print("expected outputs")
    print(T3)

    TLP2.train_online(X3, T3, eps3, maxiters=maxiters3, prints=True)

    print("\n\tCOMPUTATION")

    print("TLP computed AFTER train: \nexample: - {X} \nresult: {Y} \nexpected: {T}".format(
        X=V, Y=TLP2.process_inputs(V), T=V_T))
    print("Experimental accuracy: {}\n".format(abs(V_T-TLP2.process_inputs(V))))
    input("press 'Enter' to close")
