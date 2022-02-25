import pandas as pd 
import numpy as np 

class TreeNode():
    def __init__(self,
                X: pd.DataFrame,
                Y,
                min_samples_split = 20,
                max_depth = 5,
                depth = 0
                ) -> None:
        
        #save the inputs
        self.X = X
        if type(Y) == list:
            self.Y = np.array(Y)
        else:
            self.Y = Y
        
        #extract the features
        self.features = list(self.X.columns)
        
        #save the hyperparameters
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
        #save the depth of the node (0 --> root node)
        self.depth = depth
        
        #initialize the left and right node
        self.left = None
        self.right = None
        
        #get the mean (predicted value for the node)
        self.ymean = np.mean(Y)

        # Getting the residuals 
        self.residuals = self.Y - self.ymean
        
        #get mse
        self.mse = get_mse(Y, self.ymean)
        
        # Saving the number of observations in the node 
        self.n = len(Y)
        
        # Initiating the left and right nodes as empty nodes
        self.left = None 
        self.right = None 

        # Default values for splits
        self.best_feature = None 
        self.best_value = None 
        
    
    @staticmethod
    def get_mse(y_true, y_hat) -> float:
        return sum((y_true-y_hat)**2)/len(y_true)
    
    @staticmethod
    def get_splits(x: np.array, window: int) -> np.array:
        """
        Used in best_split
        """
        return np.convolve(x, np.ones(window), 'valid') / window
    
    def best_split(self) -> tuple:
        
        # Creating a dataset for spliting
        df = self.X.copy()
        df['Y'] = self.Y

        # Getting the GINI impurity for the base input 
        mse_base = self.mse

        # Default best feature and split
        best_feature = None
        best_value = None

        for feature in self.features:
            # Droping missing values and get the rolloing window 
            #(we try to split at every possible point of the df)
            Xdf = df.dropna().sort_values(feature)

            #obtain all the possible splitting points
            splits = self.get_splits(Xdf[feature].unique(), 2)

            for value in splits:
                # Get which observations will be in the left vs in the right node
                left_y = Xdf[Xdf[feature]<value]['Y'].values
                right_y = Xdf[Xdf[feature]>=value]['Y'].values

                # Getting the means 
                left_mean = np.mean(left_y)
                right_mean = np.mean(right_y)

                # Getting the left and right residuals 
                res_left = left_y - left_mean 
                res_right = right_y - right_mean

                # Concatenating the residuals 
                r = np.concatenate((res_left, res_right), axis=None)

                # Calculating the mse 
                n = len(r)
                r = r ** 2
                r = np.sum(r)
                mse_split = r / n

                # Checking if this is the best split so far 
                if mse_split < mse_base:
                    best_feature = feature
                    best_value = value 

                    # Setting the best gain to the current one 
                    mse_base = mse_split

        return (best_feature, best_value)
    
    def random_split(self) -> tuple:
        pass

    def grow_tree(self) -> None:
        pass
    
    def info(self) -> None:
        '''
        Print info about the tree
        '''
        pass

    def print(self) -> None:
        '''
        Print the tree in a nice way
        '''
        pass
    

