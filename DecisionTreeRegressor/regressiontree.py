import pandas as pd 
import numpy as np 

class TreeNode():
    def __init__(
        self,
        X: pd.DataFrame,
        Y: list,
        min_samples_split = None,
        max_depth = None,
        depth = None,
        node_type = None,
        rule = None
        ):        
        #save the inputs
        self.X = X
        self.Y = Y
        
        #save the hyperparameters
        self.min_samples_split = min_samples_split if min_samples_split else 20
        self.max_depth = max_depth if max_depth else 5
        
        #save the depth of the node (0 --> root node)
        self.depth = depth if depth else 0

        #save the node type
        self.node_type = node_type if node_type else 'root'
        
        #save the splitting rule
        self.rule = rule if rule else ''

        #extracting all the features
        self.features = list(self.X.columns)
        
        #get the mean (predicted value for the node)
        self.ymean = np.mean(Y)

        # Getting the residuals 
        self.residuals = self.Y - self.ymean
        
        #get mse
        self.mse = TreeNode.get_mse(Y, self.ymean)
        
        # Saving the number of observations in the node 
        self.n = len(Y)
        
        # Initiating the left and right nodes as empty nodes
        self.left = None 
        self.right = None 

        # Default values for splits
        self.best_feature = None 
        self.best_value = None 


    @staticmethod
    def get_mse(ytrue, yhat) -> float:
        """
        Method to calculate the mean squared error 
        """
        # Getting the total number of samples
        n = len(ytrue)
        # Getting the residuals 
        r = ytrue - yhat 
        # Squering the residuals 
        r = r ** 2
        # Suming 
        r = np.sum(r)
        # Getting the average and returning 
        return r / n


    @staticmethod
    def get_splits(x: np.array, window: int) -> np.array:
        """
        Used in best_split
        """
        return np.convolve(x, np.ones(window), 'valid') / window
    
    def best_split(self) -> tuple:
        
        # put in the same dataset the X and the Y
        df = self.X.copy()
        df['Y'] = self.Y

        # get the MSE at the node
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
                mse_split = np.sum(r ** 2) / len(r)

                # Checking if this is the best split so far 
                if mse_split < mse_base:
                    best_feature = feature
                    best_value = value 

                    # Setting the best gain to the current one 
                    mse_base = mse_split

        return (best_feature, best_value)
    
    def random_split(self) -> tuple:
        pass

    def fit(self) -> None:
        '''
        This method grows the tree recursively
        '''
        # put in the same dataset the X and the Y
        df = self.X.copy()
        df['Y'] = self.Y

        #till one of the stopping conditions is met we continue to split
        if (self.depth < self.max_depth) and (self.n >= self.min_samples_split):
            
            #get the best split
            best_feature, best_value = self.best_split()
            
            #when the best_feature is None it means that best_split was not able to find
            #a split for which we reduce the MSE, which means that all leaves are pure
            if best_feature is not None:
                self.best_feature = best_feature
                self.best_value = best_value

                #split the dataset we created with X and Y in left and right node
                df_left, df_right = df[df[best_feature]<=best_value].copy(), df[df[best_feature]>best_value].copy()

                #create the left and the right node
                left_node = TreeNode(
                    df_left[self.features],
                    df_left['Y'].values.tolist(),
                    depth = self.depth+1,
                    max_depth = self.max_depth,
                    min_samples_split= self.min_samples_split,
                    node_type='left node',
                    rule = f"{best_feature} <= {round(best_value,3)}"
                )

                self.left = left_node
                self.left.fit()

                right_node = TreeNode(
                    df_right[self.features],
                    df_right['Y'].values.tolist(),
                    depth = self.depth+1,
                    max_depth = self.max_depth,
                    min_samples_split= self.min_samples_split,
                    node_type='right node',
                    rule = f"{best_feature} > {round(best_value,3)}"
                )

                self.right = right_node
                self.right.fit()

    def predict_one(self, x):
        '''
        x =  row of a pd.DataFrame
        Return the predicted value according to the decision tree fitted
        It is the basis for the method predict
        The function is defined recursively
        '''
        right = self.right
        left = self.left
        yhat = self.ymean
        feature = self.best_feature
        value = self.best_value
        #check if at the node there is a split
        if right == None:
            return yhat
        else:
            if x[feature]<=value:
                print("going to the left")
                left.predict_one(x)
            else:
                print("going to the right")
                right.predict_one(x)

    def info(self, width = 4) -> None:
        '''
        Print info about the node
        '''
        #define how many spaces we need to leave depending on the depth of the node
        const = int(self.depth * width ** 1.5)
        spaces = "-" * const
        
        if self.node_type == 'root':
            print("Root")
        else:
            print(f"|{spaces} Split rule: {self.rule}")
        print(f"{' ' * const}   | MSE of the node: {round(self.mse, 2)}")
        print(f"{' ' * const}   | Count of observations in node: {self.n}")
        print(f"{' ' * const}   | Prediction of node: {round(self.ymean, 3)}")

    def print(self) -> None:
        '''
        Basic print of the three from the current node to the bottom
        '''
        #print the node
        self.info()
        
        #print all the left nodes of the tree
        if self.left is not None:
            self.left.print() 

        #print all the right nodes of the tree
        if self.right is not None:
            self.right.print()
    

