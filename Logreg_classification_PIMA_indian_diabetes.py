# Project: PIMA Indian Diabetes Classification using Logistic Regression (binary classification)
# Dataset: https://www.kaggle.com/uciml/pima-indians-diabetes-database
# Goal: 1) # Goal: to predict whether or not a given female patient will contract diabetes based on features such as BMI, age...
# 2) fine-tuning the model 

# Import the necessary modules
import pandas as pd
import numpy as np
import matplotlib.pylot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.metrics import roc_curve  # roc curve
from sklearn.metrics import roc_auc_score  # auc 
from sklearn.model_selection import GridSearchCV  # for hyperparameter tuning cross validation
from scipy.stats import randint  # for hyperparameter tuning cross validation
from sklearn.tree import DecisionTreeClassifier # for hyperparameter tuning cross validation
from sklearn.model_selection import RandomizedSearchCV  # for hyperparameter tuning cross validation


# Read the CSV file into a DataFrame: df
df = pd.read_csv('datasets/diabetes.csv')

# Create arrays for features and target variable
y = df['Outcome']
X = df.drop('Outcome')



# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



# ROC curve (also we can plot Precision-recall curve)
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)  # tpr True Positive Rate is recall 

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')  # recall
plt.title('ROC Curve')
plt.show()



# AUC 
 Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


# Hyperparameter tuning with GridSearchCV (computationally expensive)
# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)  # y = logspace(a,b,n) generates n points between decades 10^a and 10^b.
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))


# Hyperparameter tuning with RandomizedSearchCV
# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


