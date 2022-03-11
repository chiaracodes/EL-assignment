# EL Project
Team: Chiara Palma, Lauren Yoshizuka, Maria Isabel Vera Cabrera, Shwetha Salimath, Cheng Wan

### Part A

Cyberbullying classification: multiple and binary.
The code related to this part can be found in the Cyberbullying_model folder. 

Inside this folder we find 3 jupyter notebook files:

- DT_RF_Boosting_Maria: This contains all the proprocessing, feature engineering and modelling related to the following models: decision tree, random forest and boosting algorithms such as Gradient Boosting, Extreme Gradient Boosting, Adaptive Boosting and Light Gradient Boosting Machine. This was developed by Maria Isabel Vera.

- Bagging with SVM Cheng: Following Maria's proprocessing and feature engineering, this file includes an optimized Ensemble Support Vector Machines model and several untuned Tree-Based Models. 

- MultiNaiveBayes_BinarySVM_Lauren: This contains my initial work to explore and preprocess the data. I also explored simple ML algos to be later considered in ensemble methods. I looked at Multinominal Naive Bayes for the multiclassification as well as SVM for the binary classification. As for implementing ensemble methods, we saw how good Random Forest performed in the ensemble, so I tried a model that should have degraded results when using an ensemble method --> KNN.

- KNNbaggingClassifier_Lauren: clean and organized version of the KNN bagging classification ensemble method that underperformed compared to the Random Forest ensemble.

### Part B
Implementing a classification and regression algorithm for decision tree.

The dicisiontree.py we have two main class the DecisionTreeClassifier for implementing classification and the DecisionTreeRegressor for implementing the regression.
The main funtions is the decisiontree are build_tree and best_split. We get the best split by comaring the information gain after and before the split.
- For DecisionTreeClassifier we use the gini index and weight of the particular split part to calculate the information gain.
- 

The DecisionTreeExample contains examples on how to implement these classes and compared its accuracy with that of the sickit-learn library.
