import random
import pandas as pd
from decisiontree import *
from metrics import *

def train_test_split(X, Y, val = 0.2, seed = None):
    '''
    X = np.array in shape (n_samples, n_parameter)
    Y = np.array in shape (n_samples, 1) or (n_samples,)
    val = percentage of dataset for testing

    Returns X_train, X_val,Y_train, Y_val
    '''
    if seed != None:
        random.seed(int(seed))
    indexes = random.sample(range(len(X)), len(X))
    n_train = int(len(X)*(1-val))
    train_indexes = indexes[:n_train]
    test_indexes = indexes[n_train:]
    X_train, X_val,Y_train, Y_val= X[train_indexes], X[test_indexes], Y[train_indexes], Y[test_indexes]
    return X_train, X_val,Y_train, Y_val


def CVScore(model, X,Y, val = 0.2, n_splits = 5, metric = 'mse'):
    '''
    model = model of class DecisionTreeClassifier or DecisionTreeRegressor
    X = np.array in shape (n_samples, n_parameter)
    Y = np.array in shape (n_samples, 1) or (n_samples,)
    val = percentage for validation at each split
    n_splits = number of K-fold validations
    metric = metric to use to compute the score, options are ['mse', 'accuracy', 'mae', 'rmse']

    Returns the average score across the different folds and the list containing the score for each split
    '''
    metrics = []
    for _ in range(n_splits):
        X_train, X_val,Y_train, Y_val = train_test_split(X,Y, val = val)
        model.fit(X_train,Y_train)
        Y_pred = model.predict(X_val)
        metrics.append(score(Y_val, Y_pred, mode = metric))
    return np.mean(metrics), metrics


class CVGridSearch(): 
              
    def __init__(
        self,
        model: str,
        parameters: dict,
        n_splits,
        metric = 'mse',
        val = 0.2,
        verbose = 0):

        self.model = model
        self.parameters = parameters
        self.n_splits = n_splits
        self.val = val
        self.metric = metric

        #if during the fit it needs to print the results
        self.verbose = verbose

        #initialize empy table for the results
        self.result = pd.DataFrame(columns = ["min_samples_split", "max_depth", "min_score", "max_score", "sd_score", "mean_score"])

        #initialize dictionary with best parameters
        self.best = {"min_samples_split":None, "max_depth":None}
        self.best_score = None

    
    def fit(self, X, Y):
        '''
        Method of class CVGridSearch
        For each set of parameters in the dictionary it fits the model and evaluate the accuracy using CVScore
        It store the results as internal parameters of CVGridSearch
        '''
        min_samples_split = self.parameters["min_samples_split"]
        max_depth = self.parameters["max_depth"]

        results = []

        print(f"Fitting the model {len(min_samples_split)*len(max_depth)*self.n_splits} times for {len(min_samples_split)*len(max_depth)} possible parameters combination")
        
        for sample, depth in ((x,y) for x in min_samples_split for y in max_depth):
            if self.model == "DecisionTreeClassifier":
                model = DecisionTreeClassifier(min_samples_split=sample, max_depth=depth)
            else:
                model = DecisionTreeRegressor(min_samples_split=sample, max_depth=depth)
            
            _, scores = CVScore(model, X, Y, val = self.val, n_splits=self.n_splits, metric = self.metric)
            
            result = pd.DataFrame([[sample], [depth], [np.min(scores)], [np.max(scores)], [np.std(scores)],[np.mean(scores)]])
            results.append(result)
        
        self.result = pd.concat(results, axis = 1)

        if self.metric == 'accuracy':
            self.best_score = self.result.mean_score.max()

        else:
            self.best_score = self.result.mean_score.min()
        
        self.best['min_samples_split'] = self.result.loc[self.result.mean_score == self.best_score, "min_samples_split"]
        self.best['max_depth'] = self.result.loc[self.result.mean_score == self.best_score, "max_depth"]
