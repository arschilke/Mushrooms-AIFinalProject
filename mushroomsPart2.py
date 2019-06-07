import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import sys
np.set_printoptions(threshold=sys.maxsize)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV


array_str = np.loadtxt(fname="mushrooms.csv", dtype=str, delimiter=",", skiprows=1)


print(array_str[0], "\n", array_str[1], "\n", array_str[2])

print("Shape:", array_str.shape);

array = np.zeros(shape=array_str.shape, dtype=int)

le = LabelEncoder();

for i in range(array.shape[1]):
        array[:, i] = le.fit_transform(array_str[:, i])

y = array[:, 0]     # output is first col, input is rest
x = array[:, 1:]

np.savetxt("y.csv", y, delimiter=",")
np.savetxt("x.csv", x, delimiter=",")

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.99, random_state=42)

# Naive Bayes

nb = MultinomialNB(fit_prior=False)
# fit prior false, assumes uniform prior probabilities, results in higher accuracy in this case

nb.fit(X_train, y_train);
nb_train_score = nb.score(X_train, y_train)
nb_test_score = nb.score(X_test, y_test)


print("Native Bayes Train Score", nb_train_score)
print("Native Bayes Test Score", nb_test_score)

#Part 3 addtions
nb_param = {"fit_prior": (True, False), "alpha":[0.05, .1, 0.8]}

cv_nb = GridSearchCV(nb, nb_param, cv = 5)

cv_nb.fit(X_train, y_train);

cv_nb_best_params = cv_nb.best_params_
cv_nb_best_score = cv_nb.best_score_


print("CV_NB Best Params:", cv_nb_best_params)
print("CV_NB Best Score:", cv_nb_best_score)

# Multi Layered Perceptrons


mlp = MLPClassifier(max_iter=2000, learning_rate="invscaling",)

mlp.fit(X_train, y_train)

mlp_train_score = mlp.score(X_train, y_train)
mlp_test_score = mlp.score(X_test, y_test)

print("MLP Train Score", mlp_train_score)
print("MLP Test Score", mlp_test_score)

#Part 3 addtions
mlp_param = {}
cv_mlp = GridSearchCV(mlp, mlp_param, cv = 5)

cv_mlp.fit(X_train, y_train);

cv_mlp_best_params = cv_mlp.best_params_

cv_mlp_best_score = cv_mlp.best_score_


print("CV_MLP Best Params:", cv_mlp_best_params)
print("CV_MLP Best Score:", cv_mlp_best_score)



#Support Vector Machine

svm = SVC(gamma='auto')

svm.fit(X_train, y_train)

svm_train_score = svm.score(X_train, y_train)
svm_test_score = svm.score(X_test, y_test)

print("SVM Train Score", svm_train_score)
print("SVM Test Score", svm_test_score)

#Part 3 SVM addtions
svm_param = {"kernel": ( "rbf", "linear", "poly", "sigmoid"), 'C': np.arange(0.1 , 4),  "degree" : np.arange(1 , 2), "coef0" :np.arange(0 , 2), "shrinking": (True, False), "probability": (False, True), "decision_function_shape": ("ovo", "ovr") }

cv_svm = GridSearchCV(svm, svm_param, cv = 5)

cv_svm.fit(X_train, y_train)

cv_svm_best_params = cv_svm.best_params_
cv_svm_best_score = cv_svm.best_score_


print("CV_SVM Best Params:", cv_svm_best_params)
print("CV_SVM Best Score:", cv_svm_best_score)

