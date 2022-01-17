# Project: PIMA Indian Diabetes Classification using KNN (binary classification)
# Dataset: https://www.kaggle.com/uciml/pima-indians-diabetes-database
# Goal: to predict whether or not a given female patient will contract diabetes  based on features such as BMI, age, and so on

# Import necessary modules
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Read the CSV file into a DataFrame: df
df = pd.read_csv('datasets/diabetes.csv')

# Create arrays for features and target variable
y = df['Outcome']
X = df.drop('Outcome')

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
