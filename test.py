import sys
import os
import json
from randommodel import RandomModel
from rnnmodel import RNNModel

if __name__ == "__main__":
    m = RNNModel()
    try:
        train_fname = sys.argv[1]
    except:
        train_fname = os.path.join(os.getcwd(), "development.json")

    try:
        pred_fname = sys.argv[2]
    except:
        pred_fname = os.path.join(os.getcwd(), "pred.csv")
    model = m.train('training.json')
    m.predict(train_fname,pred_fname)