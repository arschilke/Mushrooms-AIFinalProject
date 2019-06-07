![Logo](https://github.com/arschilke/Mushrooms-AIFinalProject/blob/master/Picture1.svg "Logo"=100x)

# Mushrooms
### Are they edible or are they poisonous?
Project by Alyssa Schilke, Mollie Morrow, and Jennifer Moutenot
## DATASET
Mushroom Classification
- Appearance and specifications of 8,000+ different species of mushrooms (dimensions, population, color, spacing, size, number of rings, odor,  spores, edibility, etc.)
- Dimensions, population, color, spacing, size, number of rings, odor,  spores, edibility, etc.
- Originally intended to be used to determine the edibility of various mushrooms
- https://www.kaggle.com/uciml/mushroom-classification

## Problem Type 
### Supervised Learning on a Classification Problem

## Machine Learning Technique
- Our chosen methods are Multi-layer Perceptron (MLP) which is a supervised learning algorithm, and Support Vector Machine (SVM)
- MLP and SVM learn a function by training on a dataset
- MLP link to the relevant library documentation page: https://scikit-learn.org/stable/modules/neural_networks_supervised.html 
- Support Vector Machine link to the relevant library documentation page: https://scikit-learn.org/stable/modules/svm.html 
- Since we are addressing a classification problem, we will be comparing our chosen methods against Naïve Bayes
- Naïve Bayes link to the relevant library documentation page: https://scikit-learn.org/stable/modules/naive_bayes.html 
## Training/testing data split

We split the testing and training sets from the dataset randomly:
- Before: split test_size=0.75 and train_size=0.25
  - Resulted in 100% accuracy 

In order to apply cross validation (to continue with the assignment), we split the dataset at:
- train_test_split(X, y, random_state=42, test_size=0.99) and train_size=0.1
We know these are not reasonable values to set  (99% testing and 1% training) but given the circumstances of the data, these values allowed for less accuracy across the classifiers

## Data Preprocessing
In order for scikit-learn to process our data, we used LabelEncoder to convert our discrete factors into numerical values
### RESULTS FROM NAÏVE BAYES
#### Original 
- Report the training and testing errors yielded:
  - The mean accuracy of the Naive Bayes training and testing data:
    - Train score:  0.8641975308641975
    - Test score:  0.7763272410791993
#### After Cross-Validation
- Hyperparameters learned through cross-validation:
  - `nb_parameters = {'fit_prior': (True, False), 'alpha': (0.8, 0.05, 0.1, 0.5)}`
- Best score for Naive Bayes 0.8395061728395061
- Best params for Naive Bayes `{'alpha': 0.8, 'fit_prior': True}`
- After using cross validation, our test score became more accurate
This was not our best performing classifier

## RESULTS FROM MLP
#### Original
- Report the training and testing errors yielded:
  - The mean accuracy of the MLP training and testing data:
    - Train score:  1.0
    - Test score:  0.9241576526171826

#### After Cross-Validation
- Hyperparameters learned through cross-validation
  - `mlp_parameters = {'max_iter':(1000, 1200, 5000, 10000)}`
- Best score for MLP 0.9135802469135802
- Best params for MLP `{'max_iter': 1200}`
- Many hyper-parameters for MLP, default worked the most accurately
- This was our best performing classifier

## RESULTS FROM SVM
#### Original
- Report the training and testing errors yielded:
  - The mean accuracy of the SVM training and testing data:
    - Train score:  0.9876543209876543
    - Test score:  0.8977993286087281

#### After Cross-Validation
- Hyperparameters learned through cross-validation:
  - `svm_parameters = {'kernel': ('rbf', 'linear', 'poly', 'sigmoid'), 'C': (np.arange(0.1, 4)), 'degree': (np.arange(1, 2)), 'coef0': np.arange(0, 2), 'shrinking': (True, False), 'probability': (False, True), 'decision_function_shape': ('ovo', 'ovr')}`
- Best score for SVM 0.9135802469135802
- Best params for SVM `{'C': 1.1, 'coef0': 0, 'decision_function_shape': 'ovo', 'degree': 1, 'kernel': 'linear', 'probability': False, 'shrinking': True}`
- We were able to increase the accuracy using cross validation
- This was our second-best performing classifier
## Conclusion
- We found a very unusual dataset that allowed a 100% mean accuracy using MLP and SVM with the normal 25% training data settings.  
- Through our analysis of hyperparameters, we found that MLP took a very long time to cross validate due to the amount of settings to evaluate. 
- This data set only includes 21 species of mushroom, it would be interesting to see how adding other kinds of mushrooms affects the calculations. 

