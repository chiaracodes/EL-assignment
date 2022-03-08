import numpy as np

def accuracy(y_real, y_predicted):
    ''''
    y_real: np.array(n_samples, 1)
    y_predicted: np.array(n_samples,1)

    Return the accuracy
    '''
    if len(y_real)!= len(y_predicted):
        raise(ValueError("Y real and Y predicted must have the same length"))
    return np.sum(y_real == y_predicted)/len(y_real)

def mse(y_real, y_predicted):
    ''''
    y_real: np.array(n_samples, 1)
    y_predicted: np.array(n_samples,1)

    Return the mse
    '''
    if len(y_real)!= len(y_predicted):
        raise(ValueError("Y real and Y predicted must have the same length"))
    return (np.sum((y_real-y_predicted)**2))/len(y_real)

def mae(y_real, y_predicted):
    ''''
    y_real: np.array(n_samples, 1)
    y_predicted: np.array(n_samples,1)

    Return the mae
    '''
    if len(y_real)!= len(y_predicted):
        raise(ValueError("Y real and Y predicted must have the same length"))
    return np.sum(np.abs(y_real-y_predicted))/len(y_real)

def rmse(y_real, y_predicted):
    ''''
    y_real: np.array(n_samples, 1)
    y_predicted: np.array(n_samples,1)

    Return the rmse
    '''
    if len(y_real)!= len(y_predicted):
        raise(ValueError("Y real and Y predicted must have the same length"))
    return np.sqrt(np.sum((y_real-y_predicted)**2)/len(y_real))


def score(y_real, y_predicted, mode = 'accuracy'):
    #this function is used in crossvalidation in model_selection.py
    if mode == "accuracy":
        return accuracy(y_real, y_predicted)
    elif mode == "mse":
        return mse(y_real, y_predicted)
    elif mode == "mae":
        return mae(y_real, y_predicted)
    elif mode == "rmse":
        return rmse(y_real, y_predicted)
        




