import random
from decisiontree import *

def train_test_split(X, Y, val = 0.2):
    indexes = random.sample(range(len(X)), len(X))
    n_train = int(len(X)*(1-val))
    train_indexes = indexes[:n_train]
    test_indexes = indexes[n_train:]
    X_train, X_val,Y_train, Y_val= X[train_indexes], X[test_indexes], X[train_indexes], X[test_indexes]
    return X_train, X_val,Y_train, Y_val


def CVGridSearch():
    
    pass