import numpy as np

def check_sizes(y_real, y_predicted):
    if len(y_real)!= len(y_predicted):
        raise(ValueError("Y real and Y predicted must have the same length"))

    if type(y_predicted)!=np.array:
        y_predicted = np.array(y_predicted)

    if y_predicted.shape != y_real.shape:
        y_real = y_real.reshape(y_predicted.shape)
    return y_real,y_predicted

def accuracy(y_real, y_predicted):
    ''''
    y_real: np.array(n_samples, 1)
    y_predicted: np.array(n_samples,1)

    Return the accuracy
    '''
    y_real, y_predicted = check_sizes(y_real, y_predicted)

    return np.sum(y_real == y_predicted)/len(y_real)

def mse(y_real, y_predicted):
    ''''
    y_real: np.array(n_samples, 1)
    y_predicted: np.array(n_samples,1)

    Return the mse
    '''
    y_real, y_predicted = check_sizes(y_real, y_predicted)

    difference_array = np. subtract(y_real, y_predicted)
    squared_array = np. square(difference_array)
    mse = squared_array. mean()
    return mse

def mae(y_real, y_predicted):
    ''''
    y_real: np.array(n_samples, 1)
    y_predicted: np.array(n_samples,1)

    Return the mae
    '''
    y_real, y_predicted = check_sizes(y_real, y_predicted)

    return np.sum(np.abs(y_real-y_predicted))/len(y_real)

def rmse(y_real, y_predicted):
    ''''
    y_real: np.array(n_samples, 1)
    y_predicted: np.array(n_samples,1)

    Return the rmse
    '''
    y_real, y_predicted = check_sizes(y_real, y_predicted)
    
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
        




