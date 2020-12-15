
from pandas import read_csv, DataFrame as df
import numpy as np

from pnn import PNN


if __name__ == '__main__':
    np.random.seed(42)
    try:
        train_set = np.array(read_csv("../lab3/data/input/KDDTrain_procsd.csv"))
        valid_set = np.array(read_csv("../lab3/data/input/KDDTrain_20PERCENT_procsd.csv"))
        test_set = np.array(read_csv("../lab3/data/input/KDDTest_procsd.csv"))

        n = int(train_set.shape[1]-1)

        input("n = {}. Press 'Enter to start'".format(n))

        indcs = np.array([range(35000,36000), range(36000,37000), range(54000, 55000)]).reshape((3000,))

        train_set_in = train_set[indcs,0:n]
        train_set_out = train_set[indcs,n]
        
        valid_set_in = valid_set[:,0:n]
        valid_set_out = valid_set[:,n]
        
        test_set_in = test_set[:,0:n]
        test_set_out = test_set[:,n]

        pnn = PNN(train_set_in, train_set_out)
        
        print("\tRECOGNITION  (validating set)")
        valid_recog = pnn.recognize(valid_set_in).squeeze()

        print("\tRECOGNITION  (testing set)")
        test_recog = pnn.recognize(test_set_in).squeeze()

        validDF = df({
            "expected": valid_set_out,
            "recognized": valid_recog
        })
        validDF.to_csv("../lab3/data/output/KDD_validation_pnn.csv")

        testDF = df({
            "expected": test_set_out,
            "recognized": test_recog
        })
        validDF.to_csv("../lab3/data/output/KDD_testing_pnn.csv")

    except Exception as e:
        print("Exception occured:\n{}".format(e))
    finally:
        input("Press 'Enter to quit'")