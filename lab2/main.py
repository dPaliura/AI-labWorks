from numpy import array, insert

from pnn import PNN 


train_in = [[ 1,  1],
            [ 1,  2],
            [ 2,  1],
            [10, 10]]

train_out = ["A", "A", "A", "B"]

input_names = ["x1", "x2"]

test_in =  [[ 1,  0],
            [ 3,  4],
            [12, 10]]

if __name__ == "__main__":
    my_pnn = PNN(train_in, train_out, input_names)
    recognized = my_pnn.recognize(test_in)
    
    train = array(train_in, dtype=str)
    test = array(test_in, dtype=str)
    train = insert(train, len(train[0]), array(train_out), axis=1)
    test = insert(test, len(test[0]), array(recognized), axis=1)
    

    print("\tTRAINING INPUT")
    print(train)

    print("\tTEST RECOGNITION")
    print(test)
    input()
    