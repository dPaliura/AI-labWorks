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
    print("Test input:")
    train = train_in
    test = test_in
    train_in.append(train_out)
    test_in.append(recognized)
    print(train)
    print(test)
    input()
    