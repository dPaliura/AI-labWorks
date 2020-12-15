from pandas import read_csv, DataFrame as df
import numpy as np

from tlp import TLP


if __name__ == '__main__':
    np.random.seed(42)
    try:
        train_set = np.array(read_csv("data/input/KDDTrain_procsd.csv"))
        valid_set = np.array(read_csv("data/input/KDDTrain_20PERCENT_procsd.csv"))
        test_set = np.array(read_csv("data/input/KDDTest_procsd.csv"))

        n = int(train_set.shape[1]-1)
        p = int(n**.75)
        m = 1

        input("n = {}. Press 'Enter to start'".format(n))

        train_set_in = train_set[:,0:n]
        train_set_out = train_set[:,n]
        
        valid_set_in = valid_set[:,0:n]
        valid_set_out = valid_set[:,n]
        
        test_set_in = test_set[:,0:n]
        test_set_out = test_set[:,n]

        tlp = TLP(n, p, m, True)

        for i in range(1):
            print("\tTRAINING {}/{}".format(i+1, 1))
            tlp.back_prop_epoch(train_set_in, train_set_out, show_progress=True)
        
        print("\tRECOGNITION  (validating set)")
        valid_recog_1 = tlp.feed_forward(valid_set_in, True).squeeze()

        print("\tRECOGNITION  (testing set)")
        test_recog_1 = tlp.feed_forward(test_set_in, True).squeeze()

        for i in range(49):
            print("\tTRAINING {}/{}".format(i+2, 50))
            tlp.back_prop_epoch(train_set_in, train_set_out, show_progress=True)
        
        print("\tRECOGNITION  (validating set)")
        valid_recog_50 = tlp.feed_forward(valid_set_in, True).squeeze()

        print("\tRECOGNITION  (testing set)")
        test_recog_50 = tlp.feed_forward(test_set_in, True).squeeze()

        for i in range(50):
            print("\tTRAINING {}/{}".format(i+50, 100))
            tlp.back_prop_epoch(train_set_in, train_set_out, show_progress=True)
        
        print("\tRECOGNITION  (validating set)")
        valid_recog_100 = tlp.feed_forward(valid_set_in, True).squeeze()

        print("\tRECOGNITION  (testing set)")
        test_recog_100 = tlp.feed_forward(test_set_in, True).squeeze()

        print("WRITING")
        valid_set_out = valid_set_out.squeeze()
        validDF = df({
            "expected": valid_set_out,
            "recognized_1": valid_recog_1,
            "encoded_1": np.round(valid_recog_1),
            "error_1": valid_set_out - valid_recog_1,
            "recognized_50": valid_recog_50,
            "encoded_50": np.round(valid_recog_50),
            "error_50": valid_set_out - valid_recog_50,
            "recognized_100": valid_recog_100,
            "encoded_100": np.round(valid_recog_100),
            "error_100": valid_set_out - valid_recog_100,
        })
        validDF.to_csv("data/output/KDD_validation.csv", index=False)

        test_set_out = test_set_out.squeeze()
        testDF = df({
            "expected": test_set_out,
            "recognized_1": test_recog_1,
            "encoded_1": np.round(test_recog_1),
            "error_1": test_set_out - test_recog_1,
            "recognized_50": test_recog_50,
            "encoded_50": np.round(test_recog_50),
            "error_50": test_set_out - test_recog_50,
            "recognized_100": test_recog_100,
            "encoded_100": np.round(test_recog_100),
            "error_100": test_set_out - test_recog_100,
        })
        testDF.to_csv("data/output/KDD_testing.csv", index=False)

        '''
        Do the same on not supplemented KDD
        '''
        train_set = np.array(read_csv("data/input/KDDTrain_supplmntd_10perc.csv"))

        train_set_in = train_set[:,0:n]
        train_set_out = train_set[:,n]

        np.random.seed(42)
        
        tlp = TLP(n, p, m, True)

        for i in range(1):
            print("\tTRAINING {}/{}".format(i+1, 1))
            tlp.back_prop_epoch(train_set_in, train_set_out, show_progress=True)
        
        print("\tRECOGNITION  (validating set)")
        valid_recog_1 = tlp.feed_forward(valid_set_in, True).squeeze()

        print("\tRECOGNITION  (testing set)")
        test_recog_1 = tlp.feed_forward(test_set_in, True).squeeze()

        for i in range(49):
            print("\tTRAINING {}/{}".format(i+2, 50))
            tlp.back_prop_epoch(train_set_in, train_set_out, show_progress=True)
        
        print("\tRECOGNITION  (validating set)")
        valid_recog_50 = tlp.feed_forward(valid_set_in, True).squeeze()

        print("\tRECOGNITION  (testing set)")
        test_recog_50 = tlp.feed_forward(test_set_in, True).squeeze()

        for i in range(50):
            print("\tTRAINING {}/{}".format(i+50, 100))
            tlp.back_prop_epoch(train_set_in, train_set_out, show_progress=True)
        
        print("\tRECOGNITION  (validating set)")
        valid_recog_100 = tlp.feed_forward(valid_set_in, True).squeeze()

        print("\tRECOGNITION  (testing set)")
        test_recog_100 = tlp.feed_forward(test_set_in, True).squeeze()

        print("WRITING")
        valid_set_out = valid_set_out.squeeze()
        validDF = df({
            "expected": valid_set_out,
            "recognized_1": valid_recog_1,
            "encoded_1": np.round(valid_recog_1),
            "error_1": valid_set_out - valid_recog_1,
            "recognized_50": valid_recog_50,
            "encoded_50": np.round(valid_recog_50),
            "error_50": valid_set_out - valid_recog_50,
            "recognized_100": valid_recog_100,
            "encoded_100": np.round(valid_recog_100),
            "error_100": valid_set_out - valid_recog_100,
        })
        validDF.to_csv("data/output/KDD_suppl_10_percnt_validation.csv", index=False)

        test_set_out = test_set_out.squeeze()
        testDF = df({
            "expected": test_set_out,
            "recognized_1": test_recog_1,
            "encoded_1": np.round(test_recog_1),
            "error_1": test_set_out - test_recog_1,
            "recognized_50": test_recog_50,
            "encoded_50": np.round(test_recog_50),
            "error_50": test_set_out - test_recog_50,
            "recognized_100": test_recog_100,
            "encoded_100": np.round(test_recog_100),
            "error_100": test_set_out - test_recog_100,
        })
        testDF.to_csv("data/output/KDD_suppl_10_percnt_testing.csv", index=False)

    except Exception as e:
        print("Exception occured:\n{}".format(e))
    finally:
        input("Press 'Enter to quit'")